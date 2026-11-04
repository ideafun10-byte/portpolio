"""API로 받을 수 있는 CSV 데이터를 data에 저장"""

import requests


url = f"https://open.jejudatahub.net/api/proxy/bt11Daa5at1tat515DDtaa9D5aa99t11/{key}"
res = requests.get(url, params={"startDate": "200901", "endDate": "200912"}, timeout=30)
res.raise_for_status()   # 오류 응답(키 틀림 등)이면 여기서 예외 발생