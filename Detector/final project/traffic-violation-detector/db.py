from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Float, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / 'db.sqlite'

Base = declarative_base()


class Vehicle(Base):
    __tablename__ = 'vehicles'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String(64), unique=True, index=True, nullable=False)
    owner_name = Column(String(128), nullable=True)
    owner_contact = Column(String(64), nullable=True)
    # 'metadata' is a reserved attribute name on declarative bases; map column name to attribute 'meta'
    meta = Column('metadata', Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Violation(Base):
    __tablename__ = 'violations'
    id = Column(Integer, primary_key=True)
    camera_id = Column(String(64), nullable=True)
    plate_number = Column(String(64), nullable=True, index=True)
    violation_type = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=True)
    snapshot_path = Column(String(256), nullable=True)
    ocr_text = Column(String(256), nullable=True)
    challan_id = Column(Integer, ForeignKey('challans.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # specify foreign_keys to disambiguate the two-way relationship
    challan = relationship('Challan', back_populates='violation', foreign_keys='Challan.violation_id', uselist=False)


class Challan(Base):
    __tablename__ = 'challans'
    id = Column(Integer, primary_key=True)
    violation_id = Column(Integer, ForeignKey('violations.id'))
    amount = Column(Float, nullable=False)
    issued_at = Column(DateTime, default=datetime.utcnow)
    issued_by = Column(String(128), nullable=True)
    status = Column(String(32), default='issued')
    proof = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # specify foreign_keys explicitly for clarity
    violation = relationship('Violation', back_populates='challan', uselist=False, foreign_keys=[violation_id])


def get_engine(db_path: Optional[Path] = None):
    db_file = db_path or DB_PATH
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    return engine


def init_db(db_path: Optional[Path] = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    engine = engine or get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def create_violation(record: Dict[str, Any], session=None):
    engine = None
    if session is None:
        engine = init_db()
        session = get_session(engine)
    v = Violation(
        camera_id=record.get('camera_id'),
        plate_number=record.get('plate_number'),
        violation_type=record.get('violation_type'),
        confidence=record.get('confidence'),
        snapshot_path=record.get('snapshot'),
        ocr_text=record.get('ocr_text')
    )
    session.add(v)
    session.commit()
    session.refresh(v)
    return v


def create_challan(violation_id: int, amount: float, proof: Optional[str] = None, session=None):
    if session is None:
        engine = init_db()
        session = get_session(engine)
    c = Challan(violation_id=violation_id, amount=amount, proof=proof)
    session.add(c)
    session.commit()
    session.refresh(c)
    # link back: update violation
    viol = session.query(Violation).get(violation_id)
    if viol:
        viol.challan_id = c.id
        session.commit()
    return c


def get_vehicle_by_plate(plate: str, session=None):
    if not plate:
        return None
    if session is None:
        engine = init_db()
        session = get_session(engine)
    return session.query(Vehicle).filter(Vehicle.plate_number == plate).first()
