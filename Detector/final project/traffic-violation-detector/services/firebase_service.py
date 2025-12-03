import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Global flag to prevent multiple Firebase app initialization
_FIREBASE_INITIALIZED = False

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE_ADMIN = True
except ImportError:
    HAS_FIREBASE_ADMIN = False
    firebase_admin = None
    credentials = None
    firestore = None


class FirebaseService:
    """Firebase service that writes to Firestore when credentials are provided.

    Behavior:
    - If `firebase-admin` is installed and a valid service account JSON is
      available (path passed in `credentials_path` or via
      `GOOGLE_APPLICATION_CREDENTIALS`), this class initializes the Firebase
      Admin SDK and writes violations into the `violations` collection in
      Firestore.
    - Otherwise it falls back to an in-memory store so the rest of the
      application can operate without remote access.
    """

    def __init__(self, credentials_path: str = 'firebase-adminsdk.json', database_url: Optional[str] = None):
        global _FIREBASE_INITIALIZED
        
        self.credentials_path = credentials_path
        self.database_url = database_url
        self._store = {}  # in-memory store for violations
        self._firestore = None

        if not HAS_FIREBASE_ADMIN:
            logger.info("firebase_admin not installed; running in local/in-memory mode.")
            return

        creds_file = Path(credentials_path)
        # Allow GOOGLE_APPLICATION_CREDENTIALS env var to specify path too
        env_path = Path(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')) if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') else None

        cred_obj = None
        if env_path and env_path.exists():
            logger.info("Found credentials from GOOGLE_APPLICATION_CREDENTIALS: %s", env_path)
            cred_obj = env_path
        elif creds_file.exists():
            logger.info("Found credentials file at %s", creds_file)
            cred_obj = creds_file

        if cred_obj is None:
            logger.info("No credentials found; running in local/in-memory mode.")
            return

        # Initialize Firebase only once per process
        if not _FIREBASE_INITIALIZED:
            try:
                cred = credentials.Certificate(str(cred_obj))
                app_options = {}
                if self.database_url:
                    app_options['databaseURL'] = self.database_url
                firebase_admin.initialize_app(cred, app_options)
                _FIREBASE_INITIALIZED = True
                logger.info("Initialized Firebase Admin SDK successfully.")
            except Exception as e:
                logger.exception("Failed to initialize Firebase Admin SDK: %s", e)
                return

        # Get Firestore client
        try:
            self._firestore = firestore.client()
            logger.info("Connected to Firestore.")
        except Exception as e:
            logger.exception("Failed to connect to Firestore: %s", e)

    def save_violation(self, violation_data: Dict) -> str:
        """Save violation to Firestore (if available) or in-memory store.

        Returns the document id or generated id.
        """
        if not isinstance(violation_data, dict):
            raise ValueError("violation_data must be a dict")

        violation_data_copy = violation_data.copy()
        violation_data_copy.setdefault('created_at', datetime.utcnow().isoformat())

        # Prefer Firestore if initialized
        if self._firestore is not None:
            try:
                col = self._firestore.collection('violations')
                doc_ref = col.document()  # generates id
                doc_ref.set(violation_data_copy)
                vid = doc_ref.id
                logger.info("Saved violation to Firestore: %s", vid)
                return vid
            except Exception as e:
                logger.exception("Failed to save violation to Firestore: %s", e)

        # Fallback: in-memory store
        vid = 'V' + datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        self._store[vid] = violation_data_copy
        logger.debug("Saved violation to in-memory store: %s", vid)
        return vid

    def get_violation(self, violation_id: str) -> Optional[Dict]:
        """If Firestore is used, read from it; otherwise use in-memory store."""
        if self._firestore is not None:
            try:
                doc = self._firestore.collection('violations').document(violation_id).get()
                if doc.exists:
                    return doc.to_dict()
                return None
            except Exception:
                logger.exception("Failed to fetch violation %s from Firestore", violation_id)
                return None
        return self._store.get(violation_id)

    def list_violations(self):
        if self._firestore is not None:
            try:
                docs = self._firestore.collection('violations').stream()
                return [(d.id, d.to_dict()) for d in docs]
            except Exception:
                logger.exception("Failed to list violations from Firestore")
                return []
        return list(self._store.items())