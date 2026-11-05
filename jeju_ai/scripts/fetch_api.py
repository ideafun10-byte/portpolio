"""API로 받을 수 있는 CSV 데이터를 data에 저장"""

from dotenv import load_dotenv
import requests
import os

load_dotenv()
key = os.getenv("TRAVEL_KEY")


url = f"https://open.jejudatahub.net/api/proxy/bt11Daa5at1tat515DDtaa9D5aa99t11/{key}"
res = requests.get(url, params={"startDate": "200901", "endDate": "200912"}, timeout=30)


print(res.status_code,res.headers.get("Content_Type"))

