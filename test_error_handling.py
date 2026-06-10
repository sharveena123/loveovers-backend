"""Quick verification of prediction/training error handling."""
import io
import os
os.environ["FOOD_CLASSIFIER_WARMUP"] = "0"

import pandas as pd
import app as A

client = A.app.test_client()
results = []


def check(name, resp, expect_status, expect_key="error"):
    body = resp.get_json()
    ok = resp.status_code == expect_status and (expect_key is None or expect_key in (body or {}))
    results.append((name, ok, resp.status_code, body))


# ── /predict validation ──
check("predict: no body", client.post("/predict", json={}), 400)
check("predict: unknown cafe", client.post("/predict", json={"cafe_id": "nope"}), 404)

cafe_id = next(iter(A.cafe_models), None)
if cafe_id:
    items = A.cafe_models[cafe_id]["metadata"]["items"]
    check("predict: missing item", client.post("/predict", json={"cafe_id": cafe_id}), 400)
    check("predict: unknown item", client.post("/predict", json={"cafe_id": cafe_id, "item": "Unicorn Cake"}), 404)
    check("predict: bad date", client.post("/predict", json={"cafe_id": cafe_id, "item": items[0], "date": "10/06/2026"}), 400)
    r = client.post("/predict", json={"cafe_id": cafe_id, "item": items[0], "date": "2026-06-15"})
    check("predict: valid request still works", r, 200, expect_key="predicted_sales")

    check("batch: bad date", client.post("/batch-predict", json={"cafe_id": cafe_id, "date": "junk"}), 400)
    r = client.post("/batch-predict", json={"cafe_id": cafe_id, "date": "2026-06-15"})
    check("batch: valid request still works", r, 200, expect_key="predictions")
check("batch: no body", client.post("/batch-predict", json={}), 400)

# ── /assess + /train with empty / header-only CSV ──
check("assess: empty file", client.post("/assess", data={"file": (io.BytesIO(b""), "empty.csv"), "cafe_name": "T"},
                                        content_type="multipart/form-data"), 400)
check("assess: headers only", client.post("/assess", data={"file": (io.BytesIO(b"date,item,sold_qty\n"), "h.csv"), "cafe_name": "T"},
                                          content_type="multipart/form-data"), 400)
check("train: empty file", client.post("/train", data={"file": (io.BytesIO(b""), "empty.csv"),
                                                       "user_corrections": '{"date":"date","item":"item","sold_qty":"sold_qty"}'},
                                       content_type="multipart/form-data"), 400)
bad_rows = b"date,item,sold_qty\nnot-a-date,,abc\n???,,xyz\n"
check("train: all rows unusable", client.post("/train", data={"file": (io.BytesIO(bad_rows), "bad.csv"),
                                                              "user_corrections": '{"date":"date","item":"item","sold_qty":"sold_qty"}'},
                                              content_type="multipart/form-data"), 400)

# ── standardize_dataset direct ──
try:
    A.standardize_dataset(pd.DataFrame(columns=["date", "item", "sold_qty"]),
                          {"date": "date", "item": "item", "sold_qty": "sold_qty"})
    results.append(("standardize: empty df raises", False, "-", "no error raised"))
except ValueError as e:
    results.append(("standardize: empty df raises", True, "ValueError", str(e)))

mixed = pd.DataFrame({"date": ["2026-06-01", "garbage"], "item": ["Croissant", "Donut"], "sold_qty": [10, "oops"]})
cleaned = A.standardize_dataset(mixed, {"date": "date", "item": "item", "sold_qty": "sold_qty"})
results.append(("standardize: drops bad rows, keeps good", len(cleaned) == 1, "-", f"{len(cleaned)} row(s) kept"))

print()
all_ok = True
for name, ok, status, body in results:
    mark = "PASS" if ok else "FAIL"
    all_ok &= ok
    print(f"[{mark}] {name}  (status={status})")
    if not ok:
        print(f"       -> {body}")
print()
print("ALL PASSED" if all_ok else "SOME FAILED")
