"""V8.0 Render cron paper worker. Public OKX data only; NO real orders."""
import os,json
from datetime import datetime,timezone
import numpy as np,pandas as pd,requests
U=os.environ["SUPABASE_URL"].rstrip("/"); K=os.environ["SUPABASE_SECRET_KEY"]; A="burak_paper_main"
H={"User-Agent":"BurakCryptoRadarWorker/8.0","accept":"application/json"}
def okx(path,p=None):
 r=requests.get("https://www.okx.com"+path,params=p,headers=H,timeout=18);r.raise_for_status();x=r.json()
 if x.get("code")!="0":raise RuntimeError(str(x.get("msg")))
 return x.get("data",[])
def uni():
 ins=okx("/api/v5/public/instruments",{"instType":"SWAP"});ticks={x["instId"]:x for x in okx("/api/v5/market/tickers",{"instType":"SWAP"})};z=[]
 for i in ins:
  q=i.get("instId","");t=ticks.get(q)
  if i.get("state")!="live" or not q.endswith("-USDT-SWAP") or not t:continue
  try:px=float(t["last"]);v=px*float(t.get("volCcy24h") or 0)
  except:continue
  if px>0 and v>0:z.append({"id":q,"px":px,"vol":v})
 return sorted(z,key=lambda x:x["vol"],reverse=True)
def candles(q,bar="1H"):
 x=okx("/api/v5/market/candles",{"instId":q,"bar":bar,"limit":"300"});x=[r for r in x if len(r)>=9]
 d=pd.DataFrame(x,columns=["ts","open","high","low","close","volume","vc","qv","confirm"])
 for c in ["open","high","low","close","qv"]:d[c]=pd.to_numeric(d[c],errors="coerce")
 d["date"]=pd.to_datetime(pd.to_numeric(d.ts),unit="ms",utc=True)
 return d.sort_values("date").dropna(subset=["open","high","low","close"]).reset_index(drop=True)
def signal(d,sens,period):
 c=d.close.astype(float);h=d.high.astype(float);l=d.low.astype(float);prev=c.shift();tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
 atr=pd.Series(np.nan,index=d.index,dtype=float)
 if len(d)>=period:
  atr.iloc[period-1]=tr.iloc[:period].mean()
  for i in range(period,len(d)):atr.iloc[i]=(atr.iloc[i-1]*(period-1)+tr.iloc[i])/period
 stops=np.full(len(d),np.nan);buy=np.zeros(len(d),bool);sell=np.zeros(len(d),bool)
 for i in range(len(d)):
  if not np.isfinite(atr.iloc[i]):continue
  old=stops[i-1] if i and np.isfinite(stops[i-1]) else 0.;p=float(c.iloc[i]);pp=float(c.iloc[i-1]) if i else np.nan;loss=sens*float(atr.iloc[i])
  st=max(old,p-loss) if p>old and pp>old else min(old,p+loss) if p<old and pp<old else p-loss if p>old else p+loss;stops[i]=st
  if i and np.isfinite(stops[i-1]):buy[i]=p>st and pp<=old;sell[i]=p<st and pp>=old
 return (1 if buy[-1] else -1 if sell[-1] else 0)
def hdr():return {"apikey":K,"Authorization":"Bearer "+K,"Content-Type":"application/json"}
def config():
 r=requests.get(U+"/rest/v1/paper_bot_config",headers=hdr(),params={"account_id":"eq."+A},timeout=15);r.raise_for_status();x=r.json()
 if not x:raise RuntimeError("paper_bot_config missing")
 return x[0]
def config_status(c,status,msg):
 h=hdr();h["Prefer"]="resolution=merge-duplicates,return=minimal";row={**c,"account_id":A,"last_run_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),"last_status":status,"last_message":msg[:300],"updated_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}
 r=requests.post(U+"/rest/v1/paper_bot_config",headers=h,params={"on_conflict":"account_id"},json=[row],timeout=15);r.raise_for_status()
def load():
 r=requests.get(U+"/rest/v1/paper_trading_state",headers=hdr(),params={"account_id":"eq."+A},timeout=15);r.raise_for_status();x=r.json()
 if not x:raise RuntimeError("paper state missing")
 return x[0]["state"]
def save(s):
 h=hdr();h["Prefer"]="resolution=merge-duplicates,return=minimal";safe=json.loads(json.dumps(s,allow_nan=False))
 r=requests.post(U+"/rest/v1/paper_trading_state",headers=h,params={"on_conflict":"account_id"},json=[{"account_id":A,"state":safe}],timeout=15);r.raise_for_status()
def close(s,p,px,why):
 fee=p["notional"]*(px/p["entry"])*.0005;pnl=p["direction"]*p["notional"]*(px/p["entry"]-1)-p["entry_fee"]-fee
 s["cash"]+=p["margin"]+p["direction"]*p["notional"]*(px/p["entry"]-1)-fee
 s["trades"].append({"Parite":p["inst"],"Yön":"LONG" if p["direction"]==1 else "SHORT","Strateji":p["strategy"],"Risk/Ödül":p.get("reward_ratio",2),"Giriş UTC":p["time"],"Çıkış UTC":datetime.now(timezone.utc).isoformat(timespec="seconds"),"Giriş":p["entry"],"Çıkış":px,"Net P&L (USDT)":round(pnl,4),"Çıkış nedeni":why});s["positions"].remove(p)
def main():
 s=load();c=config();u=uni();quotes={x["id"]:x["px"] for x in u}
 manual=str(c.get("manual_close_inst") or "").strip()
 if manual:
  p=next((x for x in s.get("positions",[]) if x.get("inst")==manual),None);px=quotes.get(manual)
  if p is not None and px:
   close(s,p,px,"MANUEL");c["manual_close_inst"]=None
  elif p is None:
   c["manual_close_inst"]=None
 for p in list(s.get("positions",[])):
  px=quotes.get(p["inst"])
  if not px:continue
  a=px<=p["stop"] if p["direction"]==1 else px>=p["stop"];b=px>=p["target"] if p["direction"]==1 else px<=p["target"]
  if a or b:close(s,p,px,"STOP" if a else "HEDEF")
 if not bool(c.get("enabled",False)):save(s);config_status(c,"PAUSED","Worker çalıştı; yeni girişler duraklatılmış.");print("disabled; exits checked");return
 if c.get("strategy","NKRAL1")!="NKRAL1":raise RuntimeError("V8.0 cron worker currently supports NKRAL1 only")
 n=max(1,min(60,int(c.get("scan_count",30))));m=max(1,min(50,int(c.get("max_positions",5))));rr=max(2,min(5,int(c.get("reward_ratio",2))));sens=float(c.get("nk_sens",1));period=int(c.get("nk_atr",10))
 now=datetime.now(timezone.utc);today=now.strftime("%Y-%m-%d");s.setdefault("seen",[])
 if s.get("day")!=today:s["day"]=today;s["day_start"]=s["cash"]+sum(p["margin"] for p in s["positions"])
 eq=s["cash"]+sum(p["margin"]+p["direction"]*p["notional"]*(quotes.get(p["inst"],p["entry"])/p["entry"]-1) for p in s["positions"])
 if s["day_start"]-eq>=15:save(s);config_status(c,"DAILY_STOP","Günlük 15 USDT zarar eşiği; yeni giriş yok.");print("daily loss stop");return
 opened=0
 for x in u[:n]:
  if len(s["positions"])>=m or s["cash"]<10:break
  q=x["id"]
  if any(p["inst"]==q for p in s["positions"]):continue
  try:
   d=candles(q);d=d[d.confirm=="1"].reset_index(drop=True)
   if len(d)<210:continue
   bar=str(d.iloc[-1].date);key=q+"|"+bar
   if key in s["seen"]:continue
   direction=signal(d,sens,period);s["seen"].append(key)
   if not direction:continue
   # V8.2: 4H NKRAL trend confirmation. Keep R:R unchanged; only aligned 1H/4H signals may enter.
   d4=candles(q,"4H");d4=d4[d4.confirm=="1"].reset_index(drop=True)
   if len(d4)<210:continue
   trend4=signal(d4,sens,period)
   if trend4!=direction:continue
   c=d.close.astype(float);tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1);atr=float(tr.ewm(alpha=1/14,adjust=False).mean().iloc[-1]);px=x["px"]
   if not np.isfinite(atr) or atr<=0:continue
   stop=px-direction*1.5*atr;target=px+direction*1.5*rr*atr
   if stop<=0 or target<=0:continue
   frac=1.5*atr/px;eq=s["cash"]+sum(p["margin"] for p in s["positions"]);reserved=sum(p["margin"] for p in s["positions"]);avail=max(0.,eq*.8-reserved);notional=min(5/frac,eq*5*min(.16,.8/m),avail*5,s["cash"]*5*.95);margin=notional/5;fee=notional*.0005
   if margin+fee>s["cash"] or notional<10:continue
   s["cash"]-=margin+fee;s["positions"].append({"inst":q,"strategy":"NKRAL1+4H","direction":direction,"entry":px,"stop":stop,"target":target,"reward_ratio":rr,"notional":notional,"margin":margin,"entry_fee":fee,"time":now.isoformat(timespec="seconds"),"bar":bar});opened+=1
  except (ValueError,KeyError,TypeError,IndexError,requests.RequestException):continue
 s["seen"]=s["seen"][-1500:];s["last_scan"]=now.isoformat(timespec="seconds");save(s);msg=f"opened={opened} positions={len(s['positions'])}";config_status(c,"OK",msg);print("OK "+msg)
if __name__=="__main__":main()
