import os
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
default_yelp_dir = repo_root / "data" / "external" / "yelp_dataset_new"

os.environ.setdefault("SBP_YELP_DATA_DIR", str(default_yelp_dir))
os.environ.setdefault("SBP_ENABLE_LIVE_FALLBACK", "true")

from app import create_app

app=create_app()
client=app.test_client()
checks=client.get('/api/health').get_json()['checks']
print('tf_available', checks.get('tensorflow_available'))
print('tf_runtime_available', checks.get('tensorflow_runtime_available'))
print('live_ready', checks.get('live_fallback_ready'))

cand=client.get('/api/search?name=Five+Guys&limit=1').get_json()
if not cand:
    cand=client.get('/api/search?name=Grill&limit=1').get_json()
print('candidates', len(cand))
if not cand:
    raise SystemExit(0)

bid=cand[0]['business_id']
print('bid', bid)
start=time.time()
resp=client.post('/api/score', json={'business_id':bid, 'force_live_inference':True})
print('status', resp.status_code, 'seconds', round(time.time()-start,2))
out=resp.get_json()
print('availability', out.get('availability'))
print('mode', out.get('scoring_mode'))
print('reason', out.get('not_scored_reason'))
print('risk_score', out.get('risk_score'))
print('risk_bucket', out.get('risk_bucket'))
print('windows', len(out.get('recent_windows',[])))
