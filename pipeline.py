import importlib
import matplotlib.pyplot as plt
import helpers
import numpy as np
importlib.reload(helpers)

fields_to_drop = ["_STATE","FMONTH","IDATE","IMONTH",
    "IDAY","IYEAR","DISPCODE","SEQNO","_PSU",
    "CTELENUM","PVTRESD1","STATERES","CELLFON3",
    "LADULT","NUMADULT","NUMMEN","NUMWOMEN",
    "CTELNUM1","CELLFON2","CADULT","PVTRESD2",
    "CSTATE","LANDLINE","HHADULT","HLTHPLN1",
    "MARITAL","EDUCA","RENTHOM1","NUMHHOL2",
    "NUMPHON2","CPDEMO1","VETERAN3","EMPLOY1",
    "CHILDREN","INCOME2","INTERNET","MEDCOST",
    "USEEQUIP","BLIND","DECIDE", "DIFFWALK",
    "DIFFDRES","DIFFALON","SEATBELT", "IMFVPLAC", 
    "CAREGIV1","CRGVREL1","CRGVLNG1","CRGVHRS1",
    "CRGVPRB1","CRGVPERS","CRGVHOUS","CRGVEXPT",
    "VINOCRE2","VIINSUR2","CDHELP","SXORIENT", 
    "TRNSGNDR","RCSGENDR", "RCSRLTN2", "QSTVER",
    "QSTLANG","MSCODE",'_STSTR','_STRWT',
    '_RAWRAKE','_WT2RAKE','_CHISPNC','_DUALUSE',
    '_DUALCOR','_LLCPWT','_HCVU651','_DRDXAR1',
    '_PRACE1','_MRACE1','_HISPANC','_RACE',
    '_RACEG21','_RACEGR3','_RACE_G1','_CHLDCNT',
    '_EDUCAG','_INCOMG','_RFSEAT2','_RFSEAT3',
    'PHYSHLTH', 'MENTHLTH', 'BPHIGH4',
    '_CHOLCHK', 'BLOODCHO', 'CHOLCHK',
    'TOLDHI2', '_CASTHM1', '_LTASTH1',
    'ASTHMA3', 'HIVTST6', '_AGE80',
    '_AGE_G', '_LMTSCL1' , 'WEIGHT2',
    'HEIGHT3','HTIN4','HTM4','WTKG3',
    '_BMI5','_RFBMI5','QLACTLM2',
    'EXERANY2','STRENGTH','_TOTINDA',
    'STRFREQ_','PAMISS1_', '_PA150R2',
    '_PA300R2','_PA30021','_PASTRNG',
    '_PASTAE1','_LMTWRK1', 'SMOKE100',
    'USENOW3','_RFSMOK3','ALCDAY5',
    'DRNKANY5', 'DROCDY3_','_RFDRHV5',
    '_MISFRTN','_MISVEGN',
     '_FRTRESP', '_VEGRESP','_FRUITEX','_VEGETEX',
    'GENHLTH', 'FRUITJU1', 'FRUIT1', 'FVBEANS',
    'FVGREEN', 'FVORANG','VEGETAB1',
    '_FRUTSUM','_VEGESUM','_FRT16','_VEG23',
    '_VEGLT1','_FRTLT1','_AGE65YR', 'VEGEDA1_'
    ]

invalid_dict = {
    (7,9): ['GENHLTH','PERSDOC2','CHECKUP1',
            'BPHIGH4','BPMEDS','BLOODCHO',
            'CHOLCHK','TOLDHI2','CVDSTRK3',
            'ASTHMA3','CHCSCNCR','CHCOCNCR',
            'CHCCOPD1','HAVARTH3','ADDEPEV2',
            'CHCKIDNY','DIABETE3','QLACTLM2',
            'SMOKE100','SMOKDAY2','USENOW3',
            'EXERANY2','CHOLCHK','DRNKANY5',
            '_RFBMI5','_CHOLCHK','_RFSMOK3',
            'PAMISS1_','PNEUVAC3','FLUSHOT6'],
    (77,99): ['PHYSHLTH','MENTHLTH','POORHLTH',
              'DRNK3GE5','MAXDRNKS','AVEDRNK2',
              'EXRACT11'],
    (777,999): ['ALCDAY5','FRUITJU1','FRUIT1',
                'FVBEANS','FVGREEN','FVORANG',
                'VEGETAB1','EXEROFT1','EXERHMM1','MAXVO2_',
                'FC60_'],
    (7777,9999): ['WEIGHT2','HEIGHT3'],
    (9): ['_VEGLT1','_TOTINDA','_PACAT1',
          '_PAINDX1','_PA150R2', '_PA300R2',
          '_PA30021', '_PASTRNG', '_PAREC1',
          '_PASTAE1', '_FLSHOT6', '_LMTACT1',
          '_LMTWRK1','_LMTSCL1','_PNEUMO2',
          '_AIDTST3', '_RFHLTH','_RFHYPE5',
          '_LTASTH1','_CASTHM1', '_ASTHMS1',
          '_SMOKER3','_RFBING5','_RFDRHV5',
          '_FRTLT1','_RFCHOL'],
    (99): ['PAFREQ1_', 'PAFREQ2_', 'STRFREQ_'],

    (99900): ['_DRNKWEK'],
    (14): ['_AGEG5YR'],
    (99999): ['WTKG3'],
    (900): ['DROCDY3','DROCDY3_'],
    (3): ['_AGE65YR']
}

continuous_feautures = ['_DRNKWEK', 'FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_','MAXVO2_', 'FC60_'] # continuous feautures that left to winsorize
discrete_feautures = ['..']

FIRST_DROP_NAN_THRESHOLD = 70
SECOND_DROP_NAN_THRESHOLD = 30

if __name__ == "__main__":

    # Load dataset in dictionary format

    train_dict, test_dict, y_train, train_ids, test_ids = helpers.load_csv_data('data/dataset')

    # Drop feautures where FIRST_DROP_NAN_THRESHOLD % of values are nan
    train_dict, test_dict = helpers.filter_features_by_nan(train_dict,threshold=FIRST_DROP_NAN_THRESHOLD)

    # Drop "fields_to_drop"
    train_clean, test_clean= helpers.drop_and_keep_features(train_dict, test_dict, fields_to_drop)

    # Replace invalid values by nan
    helpers.replace_invalid_with_nan_inplace([train_clean,test_clean], invalid_dict)

    # Drop now feautures where SECOND_DROP_NAN_THRESHOLD % of values are nan
    train_clean, test_clean = helpers.filter_features_by_nan(train_clean,test_clean, threshold=SECOND_DROP_NAN_THRESHOLD) # Probably can keep and add missing category

                                                # --- FEAUTURE ENGINEERING --- #
    # First condition: values 1 or 2 → set to 1
    train_clean['PERSDOC2'][np.where((train_clean['PERSDOC2'] == 1) | (train_clean['PERSDOC2'] == 2))] = 1
    test_clean['PERSDOC2'][np.where((test_clean['PERSDOC2'] == 1) | (test_clean['PERSDOC2'] == 2))] = 1

    # Second condition: values 3 → set to 2
    train_clean['PERSDOC2'][np.where(train_clean['PERSDOC2'] == 3)] = 2
    test_clean['PERSDOC2'][np.where(test_clean['PERSDOC2'] == 3)] = 2

    train_clean['CHECKUP1'][np.where(
        (train_clean['CHECKUP1'] == 2) | 
        (train_clean['CHECKUP1'] == 3) | 
        (train_clean['CHECKUP1'] == 4) | 
        (train_clean['CHECKUP1'] == 8)
    )] = 2
    test_clean['CHECKUP1'][np.where(
        (test_clean['CHECKUP1'] == 2) | 
        (test_clean['CHECKUP1'] == 3) | 
        (test_clean['CHECKUP1'] == 4) | 
        (test_clean['CHECKUP1'] == 8)
    )] = 2


                                                # --- CROSS VALIDATION PREPARATION --- #

    # Cross validation: we'll test 2 approaches: **OVERSAMPLING** and **UNDERSAMPLING**.
    
    # Variables _DRNKWEK, FTJUDA1_, FRUTDA1_, BEANDAY_, GRENDAY_, ORNGDAY_ have a lot of dumb values.

    # 1st approach: winsoirze using 95 percentile and use raw data

    # 2nd approach: winsoirze using 95 percentile and use log-transformed data

    # Decide how to handle: add mising category or not? Can do first approach as baseline and then add "missing feauture" to test if works better

    # another approach: BALANCED OPTION OF LOGISTIC REGRESSION

    
                                                # --- FIRST APPROACH: CREATE EXTRA FEAUTURE FOR NANS --- #

    # categorical are provided with extra _missing
    # continuous: replaced by medians


    #helpers.winsorizor(train_clean,continuous_feautures,percentile = 95)

    #helpers.categorical_nan_filler(train_clean,continuous_feautures)

    #helpers.continuous_nan_filler(train_clean, continuous_feautures) # fill during cross validation.






