"""
모듈 ② 후보 검색

q_keys:기록 키 리스트
cat_keys:카탈로그 키 리스트
k:뽑을 갯수 

"""


from pyparsing import srange

from rapidfuzz import process, fuzz
import numpy as np

def find_cadidates(q_keys, cat_keys, k):
    korWord_i_list,korWord_array,engWord_i_list,engWord_array,starts = make_list(q_keys)


    Ktop,Kscore = serch(korWord_array, cat_keys, k,starts)
    Etop,Escore = serch(engWord_array, cat_keys, k)

    topk = np.ones((len(q_keys),k),dtype=np.int32)
    topk[korWord_i_list,:] = Ktop;
    topk[engWord_i_list,:] = Etop;

    top_scores = np.ones((len(q_keys),k),dtype=np.int32)
    top_scores[korWord_i_list,:] = Kscore;
    top_scores[engWord_i_list,:] = Escore;
    

    return topk,top_scores

def serch(q_keys, cat_keys, k,starts = None):
    all_top, all_scores = [], []

    if starts is None:
        count = len(q_keys)
    else:
        count = len(starts)-1
    for start in range(0,count,256):

        if starts is None:
            batch = q_keys[start:start+256]
            ip_array =process.cdist(batch,cat_keys,scorer = fuzz.ratio,dtype=np.uint8, workers=-1 )
        else:
            s_range = starts[start:start+257]
            batch = q_keys[s_range[0]:s_range[-1]]
            s_range = np.array(s_range) -s_range[0]
            ip_array =process.cdist(batch,cat_keys,scorer = fuzz.ratio,dtype=np.uint8, workers=-1 )
            ip_array = np.maximum.reduceat(ip_array, s_range[:-1], axis=0)
    
    
        

        n = ip_array.shape[1] #후보 배의 갯수
        k = min(k,n) #오류 제거
        topK  = np.argpartition(ip_array,n-k,axis = 1)[:,n-k:] #각 배마다 유사도 높은 배 k개씩 뽑기 
            #점수도 같이 꺼내기
        topK_scores = np.take_along_axis(ip_array,topK,axis = 1)
    
        all_top.append(topK)
        all_scores.append(topK_scores)
    
    topK = np.vstack(all_top)                               # 묶음들을 위아래로 이어 붙이기
    top_scores = np.vstack(all_scores)
    
    
    return topK,top_scores    


def make_list(q_keys):
    korWord_i_list = []
    korWord_array = []
    engWord_i_list = []
    engWord_array = []
    starts = []
    for i,v in enumerate(q_keys):
        if isinstance(v,list):
            korWord_i_list.append(i)
            starts.append(len(v))
            for i in v:
                korWord_array.append(i)
        else:
            engWord_i_list.append(i)
            engWord_array.append(v)

    for i in range(0,len(starts)-1):
        starts[i+1] = starts[i+1] + starts[i]
    starts.insert(0,0)

    return korWord_i_list,np.array(korWord_array),engWord_i_list,np.array(engWord_array),np.array(starts)