import sys,json,hashlib,pathlib,importlib.util,numpy as np,torch
from sklearn.metrics import roc_auc_score,brier_score_loss
src=pathlib.Path('/home/sandbox/work/repos/mega27-10-dl-diagnosis-suite/subitems_4_5/src/l2_transfer/run_l2_c.py')
spec=importlib.util.spec_from_file_location('l2c',src); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
p=mod.OUT; audit=p/'seed37_recovery';audit.mkdir(exist_ok=True); cp=p/'ckpt_s37_p3.pt'; sha=hashlib.sha256(cp.read_bytes()).hexdigest(); assert sha=='b2257e3f414f55c5f0274eec304d17f34fe27c1e54656563e8efa35588a9af7b'
progress=audit/'progress.json'
if progress.exists(): st=json.loads(progress.read_text())
else: st={'checkpoint_sha':sha,'stage':'val','next_i':0,'val_thr':None,'recovery_sequence':1};progress.write_text(json.dumps(st,indent=2))
assert st['checkpoint_sha']==sha
if st['stage']=='done': print(st);sys.exit()
model=mod.fresh_model(); state=torch.load(cp,weights_only=False);assert state['phase_done'];model.load_state_dict(state['model']);model.eval()
paths,y=mod.list_split(st['stage']);probfile=audit/(st['stage']+'_prob.npy'); probs=np.load(probfile).tolist() if probfile.exists() else []
# A prediction array is persisted before its cursor; recovery derives cursor from its length.
i=len(probs);assert i>=st['next_i']
for _ in range(4):
 if i>=len(paths):break
 with torch.no_grad(): a=torch.sigmoid(model(mod.load_batch(paths,list(range(i,min(i+mod.BATCH,len(paths)))))).squeeze(1)).numpy()
 probs.extend(a.tolist());i=len(probs);tmp=audit/'tmp.npy';np.save(tmp,np.asarray(probs,dtype=np.float32));tmp.replace(probfile)
 st['next_i']=i;progress.write_text(json.dumps(st,indent=2));print(st['stage'],i,'/',len(paths),flush=True)
if i==len(paths):
 arr=np.asarray(probs,dtype=np.float32)
 if st['stage']=='val':
  ths=np.linspace(.05,.95,181);st['val_thr']=float(max(ths,key=lambda t:((arr>=t)==y).mean()));st['stage']='test';st['next_i']=0;progress.write_text(json.dumps(st,indent=2));print('Threshold frozen',st['val_thr'])
 else:
  r=dict(seed=37,val_thr=st['val_thr'],test_acc=float(((arr>=st['val_thr'])==y).mean()),test_auc=float(roc_auc_score(y,arr)),test_brier=float(brier_score_loss(y,arr)))
  f=p/'l2_c_results.json';res=json.loads(f.read_text());assert all(z['seed']!=37 for z in res['results']);res['results'].append(r);accs=[z['test_acc'] for z in res['results']];res['mean_acc']=float(np.mean(accs));res['sd_acc']=float(np.std(accs,ddof=1));f.write_text(json.dumps(res,indent=2));st['stage']='done';progress.write_text(json.dumps(st,indent=2));print('EVAL',r,flush=True)
