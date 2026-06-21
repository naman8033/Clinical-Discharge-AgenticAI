"""
mock_ehr/app.py
===============

FastAPI app exposing the simulated EHR endpoints required by the
Care-Plan & EHR Validation Agent:

    GET  /ehr/patient/{patient_id}
    GET  /ehr/med_orders/{patient_id}
    GET  /ehr/labs/{patient_id}
    GET  /ehr/careplan/{patient_id}
    GET  /ehr/allergies/{patient_id}
    GET  /guidelines/pathway/{diagnosis_code}

Run:
    uvicorn mock_ehr.app:app --reload --port 8001
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import data as _ehr

app = FastAPI(
    title="Mock EHR API",
    version="1.0.0",
    description="Simulated EHR for the Agentic AI Clinical Discharge Auditor.",
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "mock-ehr"}


@app.get("/ehr/patient/{patient_id}")
def get_patient(patient_id: str):
    rec = _ehr.PATIENTS.get(patient_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return rec


@app.get("/ehr/med_orders/{patient_id}")
def get_med_orders(patient_id: str):
    if patient_id not in _ehr.PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return {"patient_id": patient_id, "med_orders": _ehr.MED_ORDERS.get(patient_id, [])}


@app.get("/ehr/labs/{patient_id}")
def get_labs(patient_id: str):
    if patient_id not in _ehr.PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return {"patient_id": patient_id, "labs": _ehr.LABS.get(patient_id, [])}


@app.get("/ehr/careplan/{patient_id}")
def get_careplan(patient_id: str):
    if patient_id not in _ehr.PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return {"patient_id": patient_id, "care_plan": _ehr.CARE_PLANS.get(patient_id, {})}


@app.get("/ehr/allergies/{patient_id}")
def get_allergies(patient_id: str):
    if patient_id not in _ehr.PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return {"patient_id": patient_id, "allergies": _ehr.ALLERGIES.get(patient_id, [])}


@app.get("/guidelines/pathway/{diagnosis_code}")
def get_pathway(diagnosis_code: str):
    rec = _ehr.GUIDELINES.get(diagnosis_code.upper())
    if not rec:
        raise HTTPException(status_code=404,
                            detail=f"No guideline pathway for code {diagnosis_code}")
    return rec
