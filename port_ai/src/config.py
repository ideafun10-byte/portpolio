"""
COLS(열 이름), 후보 수 K, 시드 같은 상수 저장 파일
"""

train_col  = {
    'TABLE_ID' : 'sample_id', 
    'VESSEL_ID' : "target_vessel_id",
    'VESSEL_NAME' : "vessel_name",
    'TOTAL_TON' : 'gross_tonnage',
    'INTERNAT_TOTAL_TON' : 'international_gross_tonnage',
    'NATION' : 'flag_name',
    'VESSEL_TYPE' : 'vessel_type',
    'ARRIVAL_YEAR' : 'arrival_year',
    'CALL_SIGN' : 'call_sign'
}

ctl_col = {
    'VESSEL_ID' : "vessel_id",
    'VESSEL_TYPE' : 'vessel_type',
    'VESSEL_NAME' : "vessel_name",
    'TOTAL_TON' : 'gross_tonnage',
    'NATION' : 'flag_name',
    'LIMITED_TON' : 'deadweight_tonnage',
    'VESSEL_SIZE' : 'size',
    'CAPACITY': 'unit'

}