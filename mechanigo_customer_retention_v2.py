# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:39:46 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import os, re, string
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar

from fuzzywuzzy import fuzz, process

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
#from gspread_formatting import *

def lev_dist(seq1, seq2):
    '''
    Calculates levenshtein distance between texts
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def import_makes():
    '''
    Import list of makes
    '''
    # output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
    #                                                                '..'))
    with open(os.getcwd() + '/makes.txt', encoding = "ISO-8859-1") as makes_file:
        makes = makes_file.readlines()
        
    makes = [re.sub('\n', '', m).strip() for m in makes]
    return makes

makes_list = import_makes()

def clean_makes(x, makes):
    '''
    Cleans carmax makes input
    
    Parameters
    ----------
    x : string
        makes string input
    makes: list of string
        list of reference makes with string datatype
    
    >>> clean_makes('PORSHE', makes_list)
    'PORSCHE'
    >>> clean_makes('PUEGEOT', makes_list)
    'PEUGEOT'
    >>> clean_makes('VW', makes_list)
    'VW'
    
    '''
    if pd.isna(x):
        return np.NaN
        
    else:
        x = x.strip().upper()
        if any((match := m) for m in makes if fuzz.partial_ratio(m, x) >= 95):
            return match
        elif process.extractOne(x, makes)[1] >= 75:
            return process.extractOne(x, makes)[0]
        else:
            return x
    

def import_models():
    '''
    Import list of makes
    '''
    with open(os.getcwd() + '/models.txt', encoding = "ISO-8859-1") as models_file:
        models = models_file.readlines()
        
    models = [re.sub('\n', '', m).strip() for m in models]
    return models

models_list = import_models()

def clean_model(model, makes, models):
    '''
    Cleans carmax model string
    
    Parameters
    ----------
    model : string
    makes : list
    * model_corrections module with module_corrections dict of regex expressions as keys
    
    Returns
    -------
    cleaned model string
    
    '''
    # uppercase and remove unnecessary spaces
    if pd.notna(model):
        model = re.sub('[(\t)(\n)]', ' ', model.upper().strip())
    
        matches = [(m, fuzz.partial_ratio(model, re.sub('(CLASS|SERIES)', '', m).strip())) for m in models if fuzz.partial_ratio(re.sub('(CLASS|SERIES)', '', m).strip(), model) >= 85]
        
        if len(matches):
            ratio_list = list(zip(*matches))[1]
            best_match = matches[ratio_list.index(max(ratio_list))][0]
            
            if ('CLASS' in best_match) or ('SERIES' in best_match):
                best_match = re.sub('(SERIES|CLASS)', '', best_match).strip()
                no_make = [re.sub(make, '', best_match).strip() for make in makes if fuzz.partial_ratio(make, best_match) >= 90]
                
                pattern = re.compile(f'(?<={no_make[0]})(\s)?[0-9]+')
                model_num = re.search(pattern, model)
                if model_num:
                    best_match = best_match + f'{model_num[0].strip()}'
                else:
                    pass
            
            if any((match_make := make) for make in makes if make in best_match):
                return re.sub(match_make, '', best_match).strip()
            else:
                return best_match.upper().strip()
            
        elif process.extractOne(model, models)[1] >= 85:
            match = process.extractOne(model, models)[0]
            
            if any((match_make := make) for make in makes if make in match):
                return re.sub(match_make, '', match).strip()
            else:
                return match.upper().strip()
        
        else:
            # proceed to askGPT
            return np.NaN
    else:
        return np.NaN

def clean_year(x):
    if pd.isna(x) or (x is None) or (x == 'None'):
        return np.NaN
    else:
        if isinstance(x, float) or isinstance(x, int):
            year = int(x)
            
        elif isinstance(x, str):
            try:
                year = int(''.join(re.findall('[0-9]', x.split('.')[0])))
            except:
                year = np.NaN
        else:
            year = np.NaN
        # if year is greater than next year, convert to 0/null
        if year > (datetime.today().year + 1):
            year = np.NaN
        else:
            pass
        
        return year

def remove_emoji(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text).strip()

def fix_name(name):
  '''
  Fix names which are duplicated.
  Ex. "John Smith John Smith"
  
  Parameters:
  -----------
  name: str
    
  Returns:
  --------
  fixed name; str
  
  '''
  name_corrections = {'MERCEDESBENZ' : 'MERCEDES-BENZ',
                      'MERCEDES BENZ' : 'MERCEDES-BENZ',
                      'IXSFORALL INC.' : 'Marvin Mendoza (iXSforAll Inc.)',
                      'GEMPESAWMENDOZA' : 'GEMPESAW-MENDOZA',
                      'MIKE ROLAND HELBES' : 'MIKE HELBES'}
  name_list = list()
  # removes emojis and ascii characters (i.e. chinese chars)
  name = remove_emoji(name).encode('ascii', 'ignore').decode()
  # split name by spaces
  for n in name.split(' '):
    if n not in name_list:
    # check each character for punctuations minus '.' and ','
        name_list.append(''.join([ch for ch in n 
                                if ch not in string.punctuation.replace('.', '')]))
    else:
        continue
  name_ = ' '.join(name_list).strip().upper()
  for corr in name_corrections.keys():
      if re.search(corr, name_):
          return name_corrections[re.search(corr, name_)[0]]
      else:
          continue
  return name_

def fix_phone_num(phone):
    '''
    Cleanup phone numbers
    
    Parameters
    ----------
    phone : string
    
    Returns:
    phone : string
    '''
    phone = ''.join(str(phone).split(' '))
    phone = ''.join(phone.split('-'))
    phone = re.sub('^\+63', '0', phone)
    phone = '0' + phone if phone[0] == '9' else phone
    return phone

def import_location():
    df_loc = pd.read_csv('ph_locations.csv')
    return df_loc 

ph_loc = import_location()

def get_best_match(query, match_list):
    if len(match_list) == 0:
        return np.NaN
    elif len(match_list) == 1:
        return match_list[0][0]
    else:
        matches, scores, ndx = list(zip(*match_list))
        if any((best_index := scores.index(s)) for s in scores if s >= 95):
            return matches[best_index]
        else:
            min_lev_dist = 100
            best_match = np.NaN
            for m in matches:
                lev_d = lev_dist(query, m)
                if lev_d < min_lev_dist:
                    best_match = m
                    min_lev_dist = lev_d
                else:
                    continue
            return best_match

def clean_location(loc, ph_loc, prov = None):
    
    city_dict = {'QC' : 'Quezon City'}
    
    
    if pd.isna(loc):
        return np.NaN, np.NaN, np.NaN
    else:
        loc = loc.title().strip()
        if ('City' in loc.split(', ')[0]) and (loc.split(', ')[0] in ph_loc[ph_loc.city.str.contains('City')]['city'].unique()):
            pass
        elif ('City' in loc.split(', ')[0]):
            loc = ', '.join([loc.split(', ')[0].split('City')[0].strip()] + loc.split(', ')[1:])
        # Check cities first
        
        if any((match := city_dict[l]) for l in city_dict.keys() if process.extractOne(l, loc.split(', '))[1] >= 85):
            city_match = match
        else:
            city_match_list = []
            for l in loc.split(', '):
                bests = process.extractBests(l, ph_loc.city)
                for b in bests:
                    if b[1] >= 75:
                        city_match_list.append(b)
            
            #city_match_list = [f[0] for f in fuzzy_city_match if f[1] >= 85]
            if len(city_match_list) > 0:
                city_match = get_best_match(loc, city_match_list)
            else:
                city_match = np.NaN
        
        if pd.notna(city_match):
            if prov is not None:
                prov_match = process.extractOne(prov, ph_loc.province)[0]
            else:
                prov_match = ph_loc[ph_loc.city == city_match]['province'].iloc[0]
                
            region_match = ph_loc[(ph_loc.city == city_match) & (ph_loc.province == prov_match)]['region'].iloc[0]
            
        else:
            if prov is not None:
                prov_match = process.extractOne(prov, ph_loc.province)[0]
            else:
                fuzzy_prov_match = process.extractBests(loc, ph_loc.province)
                prov_match_list = [f for f in fuzzy_prov_match if f[1] >= 80]
                prov_match = get_best_match(loc, prov_match_list)
            
            if pd.notna(prov_match):
                region_match = ph_loc[ph_loc.province == prov_match]['region'].iloc[0]
            
            else:
                fuzzy_region_match = process.extractBests(loc, ph_loc.region)
                region_match_list = [f for f in fuzzy_region_match if f[1] >= 85]
                region_match = get_best_match(loc, region_match_list)
        
        return city_match, prov_match, region_match

service_category_dict = {'PMS': ['(PMS|(BIRTHDAY PROMO)|(13TH MONTH PROMO)|(BONUS OFFER))',
                            'ANNUAL.*PMS PACKAGE', 'BASIC OIL CHANGE (.*)'],
                    'DISINFECTION': ['CAR.*((SANITATION (AND|&)?)|(FOGGING))', 'DISINFECT'],
                    'AIRCON': ['(AC|AIRCON) CLEANING', '(AC|AIRCON) BELT', '(AIR (AND )?|CABIN){1,2} FILTER',
                               'FREON CHARGING'],
                    'TIRES': ['VULCANIZING', '(TIRE(S)?|WHEEL) ALIGNMENT', 'TIRE INSTALLATION',
                              'TIRE BALANCING', 'TIRE ROTATION', 'TIRE MOUNTING',
                              'TIRE INFLAT(E|ION)', 'SPARE TIRE', 'TIRE(S)?',
                              'YOKOHAMA', 'GOODYEAR', 'BF(\s)?GOODRICH', 'WESTLAKE',
                              'AC(\s)DELCO(\s)?NS[0-9]0'],
                    'LIGHTS': ['LIGHT', 'BULB'],
                    'BATTERIES': ['BATTERY', 'AC(\s)?DELCO(\s)?[A-Z]{3}([- ]?[A-Z]{1})?[0-9]{2}'],
                    'BRAKES': ['BRAKE', 'CALIPER'],
                    'DIAGNOSIS & INSPECTION': ['ENGINE SCANNING', 'CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION'],
                    'MAINTENANCE': ['(OIL )?(CHANGE|REPLACE|PREMIUM PACKAGE)( OIL)?[-\s]?(GAS|DIESEL|MINERAL OIL|GEAR OIL)?(\s)?((SEMI|FULLY)(-)? SYNTHETIC|REGULAR)?(\s)?(.*)?', 
                                    'COOLANT(\s)?([0-9]L|DRAIN AND REFILL)?', 'ATF (DIALYSIS|DRAIN AND REFILL|FLUID)', 'SPARK PLUG',
                                    'DRIVE BELT', 'ENGINE FLUSHING', 'WIPER BLADE(\s)?(ASSEMBLY)?', '(IRIDIUM)?(\s)?SPARK(\s)?PLUG(S)?( OIL SEAL)?', 
                                    'COIL PACK', 'OIl FILTER', 'VORTEX PLUS ENGINE OIL ([0-9]+ LITER(S)?)',
                                    'VALVE COVER GASKET', 'CRANK(\s)?SHAFT (OIL SEAL|PULLEY)', 'TRANSMISSION PAN GASKET', 
                                    'OIL GASKET SPOOL VALVE', 'AUTOMATIC TRANSMISSION FLUSH', 'SENDING OIL SENSOR',
                                    'SHELL OIL [0-9]L', 'TOYOTA\s?(OIL)?\s?5W[0-9]0\s(GENUINE OIL)?',
                                    'HONDA (FULLY|SEMI) SYNTHETIC OIL [0-9]W[0-9]0', 'SHOCK ABSORBERS',
                                    'ALTERNATOR', 'ENGINE HOSE', 'FUEL (HOSE\s)?TANK(\sHOSE)?', 'FAN MOTOR', 
                                    'POWER STEERING PUMP KIT', 'THERMOSTAT HOUSING', 'ENGINE (FLUSHING|SUPPORT)', 
                                    'WINDSHIELD WIPER SET', 'STABILIZER LINK', 'RADIATOR FAN MOTOR', 'ABS SENSOR',
                                    'FUEL FILTER', '(TUNE|CHECK)[- ]?UP', 'FUEL PUMP', 'MAGNETIC CLUTCH', 'TINT', 'CLUTCH',
                                    'TENSIONER BEARING', 'BALL JOINT', 'DOOR HANDLE', 'STARTER ASSEMBLY', 'CVT FLUID',
                                    'MASS AIR FLOW SENSOR']
                    }

def get_service_category(x):
    for cat in service_category_dict.keys():
        for service in service_category_dict[cat]:
            if re.search(service, str(x).upper()):
                return cat
            else:
                continue
    return 'MISCELLANEOUS'

tire_brands = ["BRIDGESTONE", "DUNLOP", "GOODYEAR", "MICHELIN", "BF(\s)?GOODRICH",
               "MAXXIS", "NEXEN", "NITTO", "TOYO", "YOKOHAMA", "ALLIANCE", "ARIVO", 
               "DOUBLE(\s)?COIN", "TORQUE", "WANLI", "COOPER", "CST", "PRESA", "WESTLAKE"]


def get_service_name(service_category, service_name):
    if pd.isna(service_name):
        return np.NaN
    else:
        cat = service_category
        service_name = service_name.upper().strip()
        service = []
        # LIGHTS
        if cat == 'LIGHTS':
            lights = ['HEAD', 'TAIL', 'PARK', 'BRAKE', 'PLATE', 'SIGNAL', 'STOP', 'FOG']
            lights_list = [l for l in lights if re.search(l, service_name)]
            if len(lights_list):
                service =  f'REPLACE LIGHT BULBS ({", ".join(lights_list)})'
            else:
                service = service_name
        # BATTERY
        elif cat == 'BATTERIES':
            battery = ['REPLACE', 'CHARGING', 'NEW']
            battery_list = [b for b in battery if re.search(b, service_name)]
            if len(battery_list):
                if 'CHARGING' in battery_list:
                    service.append('BATTERY CHARGING')
                if 'REPLACE' in battery_list:
                    service.append('BATTERY REPLACEMENT')
                if 'NEW' in battery_list:
                    new_battery = re.search('((MEGA FORCE|AC(\s)?DELCO))?(\s)?NS[0-9]{2}', service_name)[0].strip()
                    service.append(f"NEW BATTERY {new_battery.upper()}")
                service = ' + '.join(service)
                if 'AC DELCO' in service:
                    service = re.sub('AC DELCO', 'ACDELCO', service)
            else:
                service = service_name
        # TIRES  
        elif cat == 'TIRES':
            tires = ['VULCANIZING', 'CHANGE SPARE TIRE', 'BALANCING', 'MOUNTING', 
                     'INSTALLATION', 'ALIGNMENT' , 'NEW']
            tires_list = [t for t in tires if re.search(t, service_name)]
            if len(tires_list):
                if "NEW" in tires_list:
                    new_tire = [re.search(tb + ".*[A-Z]*(([0-9]{3})|([0-9]+\.[0-9]*))\/[0-9]{2}\/(R)?[0-9]{2}[A-Z]?", service_name)[0].strip()
                                for tb in tire_brands if re.search(tb, service_name)]
                    try:
                        service.append(f"NEW TIRES {new_tire[0]}")
                    except:
                        service.append("NEW TIRES")
                if "CHANGE SPARE TIRE" in tires_list:
                    service.append("CHANGE SPARE TIRE")
                if "VULCANIZING" in tires_list:
                    service.append("VULCANIZING")
                t_list = [t for t in ["BALANCING", "MOUNTING", "INSTALLATION", "ALIGNMENT"] if t in tires_list]
                if len(t_list):
                    service.append(f"TIRE {'/'.join(t_list)}")
                else:
                    pass
                service = ' + '.join(service)
            else:
                service = service_name
        # AIRCON
        elif cat == "AIRCON":
            aircon_corrections = {'AC CLEANING' : 'AIRCON CLEANING'}
            aircon = ["(FREE)?(\s)?(AC|AIRCON)?(\s)?CLEAN(ING)?", "NO DASHBOARD PULLDOWN", 
                      '(AIRCON|AC) BELT', '((FREE )|(AIR AND )|(CABIN AND ))?(CABIN|AIR) FILTER(\s)?(REPLACEMENT)?',
                      'FREON CHARGING']
            aircon_list = [re.search(a, service_name)[0].strip() for a in aircon if re.search(a, service_name)]
            if len(aircon_list):
                service = ' + '.join(aircon_list)
                for aircon_key in aircon_corrections.keys():
                    if re.search(aircon_key, service):
                        service = re.sub(aircon_key, aircon_corrections[aircon_key], service)
                    else:
                        continue
            else:
                service = service_name
        
        # BRAKES
        elif cat == "BRAKES":
            brakes_corrections = {'BRAKE(S)?.*((CLEAN(ING|ER)?)|((&|AND)? ADJUST(ER)?)){1}' : 'BRAKES CLEANING AND ADJUSTING'}
            brakes = ['BRAKE(S)?.*((CLEAN(ING|ER)?)|((&|AND)? ADJUST(ER)?)){1}',
                      'BRAKE FLUID TOP[- ]?UP', 'CALIPER (PISTON|KIT (SET)?|BOLT)',
                      'HAND BRAKE CABLE', '(REAR|FRONT)?(\s)?BRAKE PAD(S)?(\s)?(REPLACEMENT)?\s?(.*)?(?=LABOR)',
                      'BRAKE(S)?.*((FLUID)|(FLUSH(ING)?)){1}', 'BRAKE(S)?.*((MASTER)|(CYLINDER|ASSEMBLY)){1}',
                      'BRAKE(S)?.*DRUM REFACING', 'BRAKE(S)?.*SHOE(S)?(REPLACEMENT)?(\s\(.*\))?', 'BRAKE HOSE']
            brakes_list = [re.search(br, service_name)[0].strip() for br in brakes if re.search(br, service_name)]
            if len(brakes_list):
                service = ' + '.join(brakes_list)
                if re.search('LABOR(\s)?(ONLY)?', service_name):
                     service += ' (LABOR)'
                     for brake_key in brakes_corrections.keys():
                         if re.search(brake_key, service):
                             service = re.sub((brake_key, brakes_corrections[brake_key], service))
                         else:
                             continue
                else:
                    pass
            else:
                service = service_name
        
        # DIAGNOSIS & INSPECTION
        elif cat == "DIAGNOSIS & INSPECTION":
            diag_corrections = {'CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION' : 'CAR DIAGNOSIS AND VISUAL INSPECTION'}
            diagnosis = ['CAR DIAGNOSIS.*(AND|&) (VISUAL)?.*INSPECTION', 'ENGINE SCANNING']
            diagnosis_list = [re.search(diag, service_name)[0].strip() for diag in diagnosis if re.search(diag, service_name)]
            if len(diagnosis_list):
                service = ' + '.join(diagnosis_list)
                for diag_key in diag_corrections.keys():
                    if re.search(diag_key, service):
                        service = re.sub(diag_key, diag_corrections[diag_key], service)
                    else:
                        continue
            else:
                service = service_name
        
        # MAINTENANCE
        elif cat == "MAINTENANCE":
            maintain = service_category_dict['MAINTENANCE']
            maintain_list = [re.search(m, service_name)[0].strip() for m in maintain if re.search(m, service_name)]
            if len(maintain_list):
                service = ' + '.join(maintain_list)
                if re.search('REPLACE', service_name):
                    service = re.sub('REPLACE', '', service)
                    service += ' REPLACEMENT'
                if re.search('LABOR(\s)?(ONLY)?', service_name):
                     service += ' (LABOR)'
            else:
                service = service_name
            
        # PMS
        elif cat == 'PMS':
            service = service_name
        
        elif cat == "DISINFECTION":
            disinfect_corrections = {'DISINFECT' : 'DISINFECTION'}
            #disinfect = ['CAR.*((SANITATION (AND|&)?)|(FOGGING))', 'DISINFECT']
            disinfect = ['SANITATION', 'FOGGING', 'DISINFECT']
            disinfect_list = [re.search(dis, service_name)[0].strip() for dis in disinfect if re.search(dis, service_name)]
            if len(disinfect_list):
                service = 'CAR ' + ' + '.join(disinfect_list)
                for disinfect_key in disinfect_corrections.keys():
                    if re.search(disinfect_key, service):
                        service = re.sub(disinfect_key, disinfect_corrections[disinfect_key], service)
                    else:
                        continue
            else:
                service = service_name
        
        elif cat == 'MISCELLANEOUS':
            service = service_name
    
        return service

def get_prior_services(x, df):
    df_temp = df[(df.date < x['date']) & (df.full_name == x['full_name']) &
                 (df.brand == x['brand']) & (df.model == x['model'])]
    df_temp.loc[:,'service_name'] = df_temp.apply(lambda x: str(x['service_name']), axis=1)
    services = ', '.join(df_temp.service_name.tolist())
    return services

def fix_address(address):
    cities = {'CALOOCAN' : 'CALOOCAN', 'LAS PIÑAS' : 'LAS PIÑAS', 
              'LAS PINAS' : 'LAS PINAS','MAKATI' : 'MAKATI', 
              'MALABON' : 'MALABON', 'MANDALUYONG' : 'MANDALUYONG', 
              'MANILA' : 'MANILA', 'MARIKINA' : 'MARIKINA', 
              'MUNTINLUPA' : 'MUNTINLUPA', 'NAVOTAS' : 'NAVOTAS', 
              'PARAÑAQUE' : 'PARAÑAQUE', 'PARANAQUE' : 'PARAÑAQUE',
              'PASAY' : 'PASAY', 'PASIG' : 'PASIG', 
              'PATEROS' : 'PATEROS', 'Q(UEZON)?.*C(ITY)?' : 'QUEZON',
              'SAN JUAN' : 'SAN JUAN', 'TAGUIG' : 'TAGUIG', 
              'VALENZUELA' : 'VALENZUELA'}
    provinces = {'RIZAL' : 'RIZAL', 'LAGUNA' : 'LAGUNA', 'BULACAN' : 'BULACAN',
                'CAVITE' : 'CAVITE' , 'TAYTAY' : 'RIZAL', 'TARLAC' : 'TARLAC',
                'NUEVA ECIJA' : 'NUEVA ECIJA', 'PAMPANGA' : 'PAMPANGA',
                'BATANGAS' : 'BATANGAS', 'QUEZON' : 'QUEZON', 'BATAAN' : 'BATAAN',
                'AURORA' : 'AURORA', 'ZAMBALES' : 'ZAMBALES'}
    
    if pd.isna(address):
        return ''
    else:
        address_clean = ''
        for city in cities.keys():
            if re.search(city, address.upper()):
                address_clean = f'{cities[city]} CITY, METRO MANILA, PHILIPPINES'
            else:
                continue
        # check if city dict has match
        if len(address_clean) == 0:
            for prov in provinces.keys():
                if re.search(prov, address.upper()):
                    address_clean = f'{provinces[prov]}, PHILIPPINES'
                else:
                    continue
        else:
            return address_clean
        # no match found
        if len(address_clean) == 0:
            address_clean = address
    
        return address_clean

@st.cache_data
def get_data():
    
    '''
    Import data from redash query
    http://app.redash.licagroup.ph/queries/103/source#114
    Perform necessary data preparations
    
    Parameters
    ----------
    None.

    Returns
    -------
    df_data: dataframe
        
    '''
    
    all_data = pd.read_csv("http://app.redash.licagroup.ph/api/queries/103/results.csv?api_key=utWqs85AqS84EsILQApE6WKhAFDaZVOlwgXdMfdW", 
                           parse_dates = ['date','appointment_date','date_confirmed','date_cancelled'])
    all_data.loc[:,'date'] = pd.to_datetime(all_data.loc[:,'date'])
    # rename columns
    all_data = all_data.rename(columns={'year': 'model_year', 
                                        'name':'status',
                                        'make': 'brand'})
    
    # filter out non integer results in "model_year"
    all_data.loc[:,'model_year'] = all_data.model_year.apply(lambda x: clean_year(x)).astype(pd.Int64Dtype())
    
    all_data.loc[:,'brand'] = all_data.brand.apply(lambda x: clean_makes(x, makes_list))
    all_data.loc[:,'model'] = all_data.apply(lambda x: '' if x.empty else fix_name(x['model']).upper(), axis=1)
    #all_data.loc[:,'model'] = all_data.model.apply(lambda x: clean_model(x, makes_list, models_list))
    all_data.loc[:,'year_month'] = all_data.apply(lambda x: str(x['date'].year) + '-' + str(x['date'].month).zfill(2), axis=1)
    
    # remove cancelled transactions
    all_data = all_data[all_data['status']!='Cancelled']
    
    # remove duplicates and fix names
    all_data.loc[:,'full_name'] = all_data.apply(lambda x: fix_name(x['full_name']).title(), axis=1)
    all_data.loc[:, 'plate_number'] = all_data.plate_number.fillna('').apply(lambda x: re.sub('[- ]', '', re.search('[a-zA-Z]+[0-9]?.*[a-zA-Z]?[0-9]+', x)[0].upper()) 
                                                                             if re.search('[a-zA-Z]+[0-9]?.*[a-zA-Z]?[0-9]+', x) else x)
    all_data.loc[:, 'phone'] = all_data.phone.apply(fix_phone_num)
    #all_data['city'], all_data['province'], all_data['region'] = all_data.address.apply(lambda x: clean_location(x, ph_loc))
    #all_data.loc[:, 'address'] = all_data.apply(lambda x: ', '.join(x['city'], x['province'], x['region']), axis=1)
    
    all_data.loc[:, 'address'] = all_data.address.apply(fix_address)
    all_data.loc[:, 'mechanic_name'] = all_data.apply(lambda x: fix_name(x['mechanic_name']).title(), axis=1)

        # desired columns
    cols = ['id', 'date', 'email','phone', 'address', 'full_name','brand', 'model', 'model_year', 
        'appointment_date', 'mechanic_name', 'category', 'service_name', 'sub_total', 'service_fee', 
        'total_cost', 'date_confirmed', 'status', 'status_of_payment','customer_id', 
        'fuel_type', 'transmission', 'plate_number','mileage', 'year_month', 'lead_source']
    
    # columns used for dropping duplicates
    drop_subset = ['full_name', 'brand', 'model', 'appointment_date', 'date']
    all_data_ = all_data[cols].drop_duplicates(subset=drop_subset, keep='first')
    
    # combine "service name" of entries with same transaction id
    all_data_.loc[:, 'service_category'] = all_data_.loc[:, 'service_name'].apply(lambda x: get_service_category(x))
    all_data_.loc[:, 'service_name'] = all_data_.apply(lambda x: get_service_name(x['service_category'], x['service_name']), axis=1)
    all_data_.loc[:, 'prior_services'] = all_data_.apply(lambda x: get_prior_services(x, all_data_), axis=1)

    df_data = all_data_.copy()
    # convert date to datetime
    df_data.loc[:,'date'] = pd.to_datetime(df_data.loc[:,'date'])

    # converts each row date to a cohort
    df_data.loc[:,'tx_month'] = df_data.apply(lambda row: row['date'].year*100 + row['date'].month, axis=1)
    # get first month & year of first purchase per full_name
    cohorts = df_data.groupby('full_name')[['date', 'tx_month']].min().reset_index()
    cohorts.columns = ['full_name', 'acq_date', 'first_acq']
    # combines cohort and first_cohort
    df_data = df_data.merge(cohorts, on='full_name', how='left')
    
    # remove test entries
    remove_entries = ['mechanigo.ph', 'frig_test', 'sample quotation']
    df_data = df_data[df_data.loc[:,'full_name'].isin(remove_entries) == False]
    
    return df_data

def get_ratio(a, b):
  return a/b if b else 999

@st.cache_data
def cohort_rfm(df, month_end_date):
    '''
    
    Parameters
    ----------
    df : dataframe
        Prepared customer transaction dataframe

    Returns
    -------
    df_retention : dataframe
        Customer dataframe with extracted RFM features

    '''
    df = df[df.date < month_end_date]
    
    df_retention = df.groupby(['full_name', 'brand', 'model']).agg(first_acq=('appointment_date', lambda x: x.min().year*100 + x.min().month),
                                       recency=('appointment_date', lambda x: (x.max() - x.min()).days),
                                       frequency=('id', lambda x: len(x) - 1),
                                       total_sales=('total_cost', lambda x: round(np.sum(x), 2)),
                                       avg_sales=('total_cost', lambda x: round(np.mean(x), 2)),
                                       T = ('appointment_date', lambda x: (month_end_date-x.min()).days + 1),
                                       year=('appointment_date', lambda x: x.min().year),
                                       month=('appointment_date', lambda x: x.min().month),
                                       month_diff=('appointment_date', lambda x: (month_end_date.year - x.max().year)*12 + (month_end_date.month - x.max().month)),
                                       last_mechanic_name=('mechanic_name', 'first'),
                                       last_txn_date = ('appointment_date', lambda x: x.max().date().strftime('%Y/%m/%d'))
                                       )
    df_retention.columns = ['first_acq', 'recency', 'frequency', 'total_sales', 
                         'avg_sales', 'T', 'year', 'month', 'month_diff', 
                         'last_mechanic_name', 'last_txn_date']
    df_retention.loc[:,'ITT'] = df_retention.apply(lambda row: round(get_ratio(row['recency'], row['frequency']), 2), axis=1)
    df_retention.loc[:, 'last_txn'] = df_retention.apply(lambda x: int(x['T'] - x['recency']), axis=1)
    df_retention = df_retention.fillna(0)
    # filter data by returning customers
    df_retention = df_retention[df_retention['avg_sales'] > 0]
    
    return df_retention


def search_for_name_retention(name, df_retention):
    '''
    Function to search for customer names in backend data
    '''
    df_retention = df_retention.reset_index()
    # lower to match with names in dataframe
    df_retention.loc[:,'full_name'] = df_retention.apply(lambda x: x['full_name'].lower(), axis=1)
    # search row with name
    names_retention = df_retention[df_retention.apply(lambda x: name.lower() in x['full_name'], axis=1)]
    df_temp_retention = names_retention[['full_name', 'phone', 'address', 'brand', 'model', 'model_year', 'plate_number', 
                                         'service_name', 'prior_services', 'last_mechanic_name', 'frequency', 'ITT', 'last_txn', 
                                         'last_txn_date', 'month_diff', 'avg_sales', 'total_sales']]
    df_temp_retention.loc[:, 'full_name'] = df_temp_retention.loc[:, 'full_name'].str.title()
    # round off all columns except cohort
    round_cols = ['avg_sales', 'last_txn', 'ITT', 'total_sales']
    df_temp_retention.loc[:, round_cols] = df_temp_retention.loc[:, round_cols].round(3)
    #df_temp_retention.loc[:, 'cohort'] = df_temp_retention.loc[:, 'cohort'].apply(int)
    df_temp_retention = df_temp_retention.set_index('full_name')
    return df_temp_retention

def combine_customer_data(df_data, df_retention):
    '''
    Combine dataframes for display/filter-ready use
    '''
    df_temp = df_data.reset_index()[['full_name', 'phone', 'email', 'address', 'brand', 'model', 
                                     'model_year', 'plate_number', 'service_name', 
                                     'prior_services','tx_month', 'lead_source']]\
                                    .drop_duplicates(subset=['full_name', 'brand', 'model'], keep='first')
    
    df_temp_ret = df_retention[['last_mechanic_name', 'frequency', 'ITT', 'last_txn', 'last_txn_date', 'first_acq',  
                                'total_sales', 'avg_sales', 'month_diff']].reset_index()
    
    df_merged = pd.merge(df_temp, df_temp_ret, how='right', left_on=['full_name', 'brand', 'model'], 
                                                            right_on=['full_name', 'brand', 'model']).dropna()
    # Capitalize first letter of each name
    df_merged.loc[:, 'full_name'] = df_merged.loc[:, 'full_name'].str.title()
    return df_merged

def customer_search(df_data, df_retention):
    '''
    Displays retention info of selected customers.

    Parameters
    ----------
    df_data : dataframe
    df_retention : dataframe
    models : list
        list of fitted Pareto/NBD and Gamma Gamma function

    Returns
    -------
    df_retention : dataframe
        df_retention with updated values

    '''
    # Reprocess dataframe entries to be displayed
    df_merged = combine_customer_data(df_data, df_retention)
    
    # table settings
    df_display = df_merged.sort_values(by='full_name')
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column('full_name', headerCheckboxSelection = True)
    gridOptions = gb.build()
    
    # selection settings
    data_selection = AgGrid(
        df_display,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        autoSizeColumn = 'full_name',
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height=400, 
        reload_data=False)
    
    selected = data_selection['selected_rows']
    
    if selected:           
        # row/s are selected
        
        df_list_retention = [search_for_name_retention(selected[checked_items]['full_name'], df_merged) 
                             for checked_items in range(len(selected))]
        
        df_list_retention = pd.concat(df_list_retention)
        st.dataframe(df_list_retention)          

    else:
        st.write('Click on an entry in the table to display customer data.')
        df_list_retention = pd.DataFrame()
        
    return df_list_retention

def write_gsheet(df, sheet_name, gsheet_key):
    '''
    Creates new sheet in designated googlesheet and writes selected data from df
    
    Parameters
    ----------
    df: dataframe
        dataframe to write to google sheet
    
    '''
    credentials = {
      "type": "service_account",
      "project_id": "xenon-point-351408",
      "private_key_id": "f19cf14da43b38064c5d74ba53e2c652dba8cbfd",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC5fe2N4yS74jTP\njiyv1EYA+XgnrTkZHwMx4ZY+zLuxx/ODPGxJ3m2e6QRUtz6yBUp1DD3nvzaMYY2d\nea6ti0fO2EPmmNIAZzgWVMOqaGePfXZPN1YN5ncLegZFheZuDrsz0/E+KCVUpLbr\nWBRTBF7l0sZ7paXZsVYOu/QAJI1jPRNF3lFUxMDSE8eGx+/oUmomtl+NfCi/FEJ5\nFCU4pF1FQNmVo885HGe9Tx7UywgaXRvAGJZlA4WVie4d5Jhj8LjZRhSH8+uDgdGX\ngc/4GI8U831oQ2lHsrtYIHHNzs1EG/8Ju+INdgR/Zc5SxNx/BSF8gV7kSueEd8+/\nXlobf5JZAgMBAAECggEAHRPWBOuKCx/jOnQloiyLCsUQplubu0nmxM+Br3eFptFa\n5YQ3z36cPZB2mtcc72gv61hPbgBGC0yRmBGGpfLS/2RchI4JQYHsw2dnQtPaBB7d\nSH66sTQjDjwDNqvOWwtZIj9DroQ5keK+P/dPPFJPlARuE9z8Ojt365hgIBOazGb2\ngIh9wLXrVq7Ki8OXI+/McrxkH3tDksVH2LmzKGtWBA56MRY0v9vnJFjVd+l8Q+05\nIw4lQXt55dK7EmRLIfLnawHYIvnpalCWPe6uAmCTeoOuGASLFJJR2uzcOW9IxM0a\nMkR2dduu5vQl/ahJwxZ2cH40QJUdy7ECQg5QG4qL1wKBgQDugyaPEdoUCGC6MUas\nFR4kwDIkHj/UkgzYtsemmGG0rXCqVtIerPd6FvtKlN8BDzQbyqCaw/pDUqjFoGXN\nW969vkN5Uj9YaQ5qV8c9WLbCcMw9gT6rvqyC8b8FgwaWMKHx7TgI/8xXQ666XqpT\nMTAfINWWei0e/Scqqu6hw0v+UwKBgQDHF5ce9y9mHdVb8B7m0Oz4QIHksktKfoQa\nLoGS601zK6Rr6GeEHb03s4KLG5q9L/o9HUTXqyKERnofdEdfsGsnrKbz2Wsnr8Mk\nGwnNcPTvI3uYkeTBS4paNUxZyGVbxDOrRbBYukgwacaUIGbZ5+we1BxlVN04+l5W\nvAlNEvlfIwKBgBWMcdJhOYOv0hVgWFM5wTRuzNjohrnMzC5ULSuG/uTU+qXZHDi7\nRcyZAPEXDCLLXdjY8LOq2xR0Bl18hVYNY81ewDfYz3JMY4oGDjEjr7dXe4xe/euE\nWY+nCawUz2aIVElINlTRz4Ne0Q1zeg30FrXpQILM3QC8vGolcVPaEiaTAoGBALj7\nNjJTQPsEZSUTKeMT49mVNhsjfcktW9hntYSolEGaHx8TxHqAlzqV04kkkNWPKlZ2\nR2yLWXrFcNqg02AZLraiOE0BigpJyGpXpPf5J9q5gTD0/TKL2XSPaO1SwLpOxiMw\nkPUfv8sbvKIMqQN19XF/axLLkvBJ0DWOaKXwJzs5AoGAbO2BfPYQke9K1UhvX4Y5\nbpj6gMzaz/aeWKoC1KHijEZrY3P58I1Tt1JtZUAR+TtjpIiDY5D2etVLaLeL0K0p\nrti40epyx1RGo76MI01w+rgeZ95rmkUb9BJ3bG5WBrbrvMIHPnU+q6XOqrBij3pF\nWQAQ7pYkm/VubZlsFDMvMuA=\n-----END PRIVATE KEY-----\n",
      "client_email": "googlesheetsarvin@xenon-point-351408.iam.gserviceaccount.com",
      "client_id": "108653350174528163497",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/googlesheetsarvin%40xenon-point-351408.iam.gserviceaccount.com"
    }
    
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(gsheet_key)
    
    new_sheet_name = sheet_name
    r,c = df.shape
    
    try:
        sh.add_worksheet(title=new_sheet_name,rows = r+1, cols = c+1)
        worksheet = sh.worksheet(new_sheet_name)
    except:
        worksheet = sh.worksheet(new_sheet_name)
        worksheet.clear()
    worksheet.update([df.columns.tolist()]+df.values.tolist())

def read_gsheet(url, title):
    
    credentials = {
      "type": "service_account",
      "project_id": "xenon-point-351408",
      "private_key_id": "f19cf14da43b38064c5d74ba53e2c652dba8cbfd",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC5fe2N4yS74jTP\njiyv1EYA+XgnrTkZHwMx4ZY+zLuxx/ODPGxJ3m2e6QRUtz6yBUp1DD3nvzaMYY2d\nea6ti0fO2EPmmNIAZzgWVMOqaGePfXZPN1YN5ncLegZFheZuDrsz0/E+KCVUpLbr\nWBRTBF7l0sZ7paXZsVYOu/QAJI1jPRNF3lFUxMDSE8eGx+/oUmomtl+NfCi/FEJ5\nFCU4pF1FQNmVo885HGe9Tx7UywgaXRvAGJZlA4WVie4d5Jhj8LjZRhSH8+uDgdGX\ngc/4GI8U831oQ2lHsrtYIHHNzs1EG/8Ju+INdgR/Zc5SxNx/BSF8gV7kSueEd8+/\nXlobf5JZAgMBAAECggEAHRPWBOuKCx/jOnQloiyLCsUQplubu0nmxM+Br3eFptFa\n5YQ3z36cPZB2mtcc72gv61hPbgBGC0yRmBGGpfLS/2RchI4JQYHsw2dnQtPaBB7d\nSH66sTQjDjwDNqvOWwtZIj9DroQ5keK+P/dPPFJPlARuE9z8Ojt365hgIBOazGb2\ngIh9wLXrVq7Ki8OXI+/McrxkH3tDksVH2LmzKGtWBA56MRY0v9vnJFjVd+l8Q+05\nIw4lQXt55dK7EmRLIfLnawHYIvnpalCWPe6uAmCTeoOuGASLFJJR2uzcOW9IxM0a\nMkR2dduu5vQl/ahJwxZ2cH40QJUdy7ECQg5QG4qL1wKBgQDugyaPEdoUCGC6MUas\nFR4kwDIkHj/UkgzYtsemmGG0rXCqVtIerPd6FvtKlN8BDzQbyqCaw/pDUqjFoGXN\nW969vkN5Uj9YaQ5qV8c9WLbCcMw9gT6rvqyC8b8FgwaWMKHx7TgI/8xXQ666XqpT\nMTAfINWWei0e/Scqqu6hw0v+UwKBgQDHF5ce9y9mHdVb8B7m0Oz4QIHksktKfoQa\nLoGS601zK6Rr6GeEHb03s4KLG5q9L/o9HUTXqyKERnofdEdfsGsnrKbz2Wsnr8Mk\nGwnNcPTvI3uYkeTBS4paNUxZyGVbxDOrRbBYukgwacaUIGbZ5+we1BxlVN04+l5W\nvAlNEvlfIwKBgBWMcdJhOYOv0hVgWFM5wTRuzNjohrnMzC5ULSuG/uTU+qXZHDi7\nRcyZAPEXDCLLXdjY8LOq2xR0Bl18hVYNY81ewDfYz3JMY4oGDjEjr7dXe4xe/euE\nWY+nCawUz2aIVElINlTRz4Ne0Q1zeg30FrXpQILM3QC8vGolcVPaEiaTAoGBALj7\nNjJTQPsEZSUTKeMT49mVNhsjfcktW9hntYSolEGaHx8TxHqAlzqV04kkkNWPKlZ2\nR2yLWXrFcNqg02AZLraiOE0BigpJyGpXpPf5J9q5gTD0/TKL2XSPaO1SwLpOxiMw\nkPUfv8sbvKIMqQN19XF/axLLkvBJ0DWOaKXwJzs5AoGAbO2BfPYQke9K1UhvX4Y5\nbpj6gMzaz/aeWKoC1KHijEZrY3P58I1Tt1JtZUAR+TtjpIiDY5D2etVLaLeL0K0p\nrti40epyx1RGo76MI01w+rgeZ95rmkUb9BJ3bG5WBrbrvMIHPnU+q6XOqrBij3pF\nWQAQ7pYkm/VubZlsFDMvMuA=\n-----END PRIVATE KEY-----\n",
      "client_email": "googlesheetsarvin@xenon-point-351408.iam.gserviceaccount.com",
      "client_id": "108653350174528163497",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/googlesheetsarvin%40xenon-point-351408.iam.gserviceaccount.com"
    }
    
    gsheet_key = re.search('(?<=\/d\/).*(?=\/edit)', url)[0]
    gc = gspread.service_account_from_dict(credentials)
    wb = gc.open_by_key(gsheet_key)
    
    try:
        sh = wb.worksheet(title)
        df = pd.DataFrame.from_records(sh.get_all_records())
    except:
        df = None
    
    return df

def prep_gsheet(df):
    df_gsh = df.copy()
    
    new_cols = ['date_messaged', 'remarket', 'date_followup', 'followup_remarket',
                'engagement', 'booking_date', 'appointment_date']
    for new_col in new_cols:
        df_gsh.loc[:, new_col] = ''
    
    df_gsh.loc[:, 'model_year'] = df_gsh.loc[:, 'model_year'].astype(str)
    df_gsh.loc[:, 'vehicle'] = df_gsh[['model_year', 'brand', 'model']]\
        .apply(lambda x: x.str.cat(sep = ' ',na_rep = '').strip().upper(), axis=1)
    
    cols = ['month_diff', 'full_name', 'phone', 'email', 'address'] + new_cols +\
        ['vehicle', 'plate_number', 'service_name', 'prior_services', 'tx_month', 'lead_source', 'last_mechanic_name',
         'frequency', 'ITT', 'last_txn', 'last_txn_date', 'first_acq', 'total_sales',
         'avg_sales']
    
    df_gsh = df_gsh[cols].rename(columns = {'phone' : 'contact_number',
                                      'email' : 'email_address',
                                      'service_name' : 'service',
                                      'tx_month' : 'transaction_month',
                                      'last_mechanic_name' : 'last_tech_assigned',
                                      'frequency' : 'returns',
                                      'last_txn' : 'last_transaction',
                                      'last_txn_date' : 'last_transaction_date',
                                      },
                           )
    
    df_gsh.columns = [' '.join(col.split('_')).upper() for col in df_gsh.columns]
    
    return df_gsh.sort_values(by = ['MONTH DIFF', 'FULL NAME'], ascending = True)

def get_url(key):
    key_url = 'https://docs.google.com/spreadsheets/d/1umX77-c-XIu649ZTjeaZIjtFTd-WXVsDOjL_uVOLhZM/edit#gid=0'
    key_sheet = read_gsheet(key_url, 'url')
    
    if key in key_sheet.month_year.unique():
        return key_sheet[key_sheet.month_year == key]['url'].values[0]
    else:
        return None

def add_url(key, value):
    key_url = 'https://docs.google.com/spreadsheets/d/1umX77-c-XIu649ZTjeaZIjtFTd-WXVsDOjL_uVOLhZM/edit#gid=0'
    key_sheet = read_gsheet(key_url, 'url').set_index('month_year')
    
    key_sheet.loc[key, 'url'] = value
    key_sheet = key_sheet.reset_index()
    gsheet_key = re.search('(?<=\/d\/).*(?=\/edit)', key_url)[0]
    write_gsheet(key_sheet, 'url', gsheet_key)
    

def write_retention_data(data, write_url):
    # 6-7 months due
    due_6_7 = data[data.month_diff.isin([6,7]) & data.frequency.isin([0,1])]
    
    # 8-12 months_due
    due_8_12 = data[data.month_diff.between(8,12) & data.frequency.isin([0,1])]
    
    # churned / 12+ months
    churned = data[data.month_diff > 12]
    
    #write_url = 'https://docs.google.com/spreadsheets/d/1_Lxyx0hhK-jwpigGEbNP2YFd43FYOeKTDze_N9Sl_XQ/edit#gid=1750212761'
    write_key = re.search('(?<=\/d\/).*(?=\/edit)', write_url)[0]
    
    write_gsheet(df_merged, 'Masterlist', write_key)
    write_gsheet(prep_gsheet(due_6_7), '6-7 MOS DUE', write_key)
    write_gsheet(prep_gsheet(due_8_12), '8-12 MOS DUE', write_key)
    write_gsheet(prep_gsheet(churned), 'CHURNED', write_key)


def retention_charts(read_url, sheet_name, month_start_date, month_end_date):
    df_temp = read_gsheet(read_url, sheet_name).copy()
    st.subheader('MESSAGE TRACKING')
    month_days = pd.date_range(start = month_start_date, 
                               end = month_end_date)
    dct = {}
    df_temp['DATE MESSAGED'] = df_temp['DATE MESSAGED'].replace('-', np.NaN)
    ref = pd.to_datetime(df_temp['DATE MESSAGED']).value_counts()
    for day in month_days:
        dct[day.strftime('%Y-%m-%d')] = ref[day] if day in ref.index else 0
    date_messaged = pd.DataFrame(data = dct.values(), 
                                 index = dct.keys())
    date_messaged.columns = ['count']
    # bar chart
    st.bar_chart(data = date_messaged)
    # metric
    st.metric('TOTAL MESSAGED',
              value = str(date_messaged['count'].sum()) + \
                  ' (' + str(round(date_messaged['count'].sum()*100/len(df_temp), 2)) + ' %)')
    
    st.subheader('ENGAGEMENT')
    engagement = df_temp[df_temp['DATE MESSAGED'] != ''].loc[:, 'ENGAGEMENT'].replace('', '-').value_counts()
    labels = engagement.index
    sizes = engagement.values
    percent = [str(round(pct, 2)) + ' %' for pct in 100.*sizes/sizes.sum()]
    explode = np.linspace(0, 0.1*(len(sizes)-1), len(sizes))
    legend = ['{0} - {1:1d}'.format(i,j) for i,j in zip(labels, sizes)]
    
    fig, ax = plt.subplots()
    engagement.plot(kind = 'pie', 
                    labels = percent, 
                    explode = explode,
                    startangle = 90,
                    ax = ax)
    ax.axis('equal')
    ax.legend(legend, loc = 'center left', bbox_to_anchor=(-0.1, 1.))
    st.pyplot(fig)
    

def show_retention_data(read_url, month_start_date, month_end_date):
    #read_url = 'https://docs.google.com/spreadsheets/d/1tyvgjTOQu0LZc4lNblItvKnJrCvv9VsFmDkCahqHg-8/edit#gid=926286274'
    #read_url = get_url(month_year)
    #read_key = re.search('(?<=\/d\/).*(?=\/edit)', read_url)[0]
    #df_67 = read_gsheet(read_key, 'DUE 6-7 MOs')
    
    # date_messaged
    
    with st.expander('6-7 MOS DUE', expanded = False):
        retention_charts(read_url, '6-7 MOS DUE', month_start_date,
                         month_end_date)
    
    with st.expander('8-12 MOS DUE', expanded = False):
        retention_charts(read_url, '8-12 MOS DUE', month_start_date,
                         month_end_date)
    
    with st.expander('CHURNED', expanded = False):
        retention_charts(read_url, 'CHURNED', month_start_date,
                         month_end_date)

@st.cache_data
def calc_retention_rate(filtered_df, date_range, rate_or_actual):
    '''
    Calculates retention rates for non-first acq cohorts
    
    Parameters
    ----------
    filtered_df : dataframe
        filtered df_data
    date_range: list
        list of int year_month ("%Y%m" or "202201") Note leading zero fill via zfill
    rate_or_actual: str
        string with values of "Rate" or "Actual"
    
    Returns
    -------
    retention_rate_dict: dictionary
        dictionary with date range values as keys and values with list of retention rates per month
    
    '''
    
    retention_rate_dict = {}
    for date_index, date_val in enumerate(date_range):
        unique_users_pool = list(filtered_df[filtered_df['tx_month']==date_val]['full_name'].unique())
        retention_rate = []
        for year_month_index, year_month in enumerate(date_range[date_index:]):
            if year_month_index == 0:
                # get users with more than 1 transaction in same month
                users = filtered_df[filtered_df['tx_month'] == year_month]['full_name'].value_counts()
                curr_unique_users = list(users[users > 1].index)
            else:
                curr_unique_users = list(filtered_df[filtered_df['tx_month']==year_month]['full_name'].unique())
            # curr_unique_users = list((filtered_df[filtered_df['tx_month] == year_month]['full_name'].value_counts() > 1).index)
            # XOR / not intersection
            # new_unique_users = list(set(unique_users_pool) ^ set(curr_unique_users))
            try:
                if rate_or_actual == "Rate":
                    retention_rate.append(round((len(list(set(curr_unique_users) & set(unique_users_pool))))/len(unique_users_pool), 3))
                else:
                    retention_rate.append(len(list(set(curr_unique_users) & set(unique_users_pool))))
            except:
                retention_rate.append(0)
        if rate_or_actual == 'Rate':
            retention_rate.insert(0, 1)
        else:
            retention_rate.insert(0, len(unique_users_pool))
        retention_rate.extend([np.nan]*(len(date_range) - len(retention_rate)))
        retention_rate_dict[date_val] = retention_rate
    return retention_rate_dict
    
def plot_cohort_analysis(df, column_name, value, start_date, rate_or_actual):
    '''
    Calculate and plot return rate for chosen cohort
    '''
    # setup date_range of data and chart
    date_range = pd.date_range(start = datetime(start_date.year, start_date.month, 1), 
                                 end = datetime(datetime.today().year, 
                                                   datetime.today().month, 1)\
                                                 + timedelta(days=30), 
                                 freq='1M')\
                                .strftime('%Y%m').tolist()
    date_range = [int(d) for d in date_range]
    # filter data
    if column_name == "first_acq":
        #cohort_pivot = overall_cohort_analysis(df, start_date, rate_or_actual)
        headers = sorted(list(df['first_acq'].unique()))
        start_month = int(str(start_date.year) + str(start_date.month).zfill(2))
        filtered_df = df[df['first_acq'].isin(list(headers[headers.index(start_month):]))]

    elif column_name == 'mechanic_name':
        customers = df[(df.mechanic_name == value) &
                       (df.date.dt.date >= start_date)]['full_name'].unique()
        filtered_df = df[df.full_name.isin(customers)]
        
    elif column_name == 'lead_source':
        temp_df = df[~(df.lead_source.isnull())]
        if value == 'website_website':
            customers = temp_df[(temp_df.lead_source != 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers)) & (temp_df.lead_source != 'backend')]
        # use first acq and tx_month
        elif value == 'website_backend':
            customers = temp_df[(temp_df.lead_source != 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend') & (temp_df.date.dt.date > temp_df.acq_date)) | 
                                  (temp_df.full_name.isin(customers) & (temp_df.lead_source != 'backend') & (temp_df.date.dt.date == temp_df.acq_date))]
        elif value == 'backend_website':
            customers = temp_df[(temp_df.lead_source == 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[(temp_df.full_name.isin(customers) & (temp_df.lead_source != 'backend') & (temp_df.date.dt.date > temp_df.acq_date)) | 
                                  (temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend') & (temp_df.date.dt.date == temp_df.acq_date))]
        elif value == 'backend_backend':
            customers = temp_df[(temp_df.lead_source == 'backend') & (temp_df.date.dt.date >= start_date) & (temp_df.date.dt.date == temp_df.acq_date)]['full_name'].unique()
            filtered_df = temp_df[temp_df.full_name.isin(customers) & (temp_df.lead_source == 'backend')]
    else:
        filtered_df = df[df[column_name]==value]
        
    # calculate return rate for selected category value
    cohort_dicts = calc_retention_rate(filtered_df, date_range, rate_or_actual)
    cohort_pivot = pd.DataFrame.from_dict(cohort_dicts, orient='index')
    cohort_pivot.columns = range(-1, len(date_range))
    
    fig_dims = (16, 16)
    fig, ax = plt.subplots(figsize=fig_dims)
    #ax.set(xlabel='Months After First Purchase', ylabel='First Purchase Cohort', title="Cohort Analysis")
    y_labels = [year_month for year_month in date_range]
    x_labels = np.array(list(range(0, len(y_labels))))-1
    plt.yticks(ticks=range(len(date_range)), labels=y_labels, fontsize=15, rotation=90)
    plt.xticks(x_labels, x_labels, fontsize=15)
    # adjusted scale for colorbar via vmin/vmax
    sns.heatmap(cohort_pivot, annot=True, fmt= '.1%' if rate_or_actual=="Rate" else '.0f' , mask=cohort_pivot.isnull(), 
                square=True, linewidths=.5, cmap=sns.cubehelix_palette(8), annot_kws={"fontsize":11},
                vmin=0, vmax=0.1 if rate_or_actual=="Rate" else np.max(np.max(cohort_pivot.iloc[:,1:])))
    
    plt.xlabel('Months After Acquisition', size=18)
    plt.ylabel('Acquisition Month', size=18)
    plt.title('Cohort Analysis')
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    st.dataframe(filtered_df[['date', 'full_name', 'phone', 'lead_source', 'brand', 'model', 'model_year', 'mechanic_name', 'service_name', 'acq_date']])



## =========================== main flow ======================================
if __name__ == '__main__':
    st.title('MechaniGO.ph Customer Retention Tool')
    # import data and preparation
    df_data = get_data()
    
    # select month & year to get customer returnees
    years = ['2024', '2023', '2022']
    mo_col, yr_col = st.columns(2)
    with mo_col:
        month = st.selectbox('Month:',
                             options = list(calendar.month_abbr)[1:],
                             index = datetime.today().month - 1)
    with yr_col:
        year = st.selectbox('Year:',
                            options = years,
                            index = 0)
    
    # calculates month ranges
    selected_month_len = calendar.monthrange(int(year), list(calendar.month_abbr).index(month))
    month_start_date = datetime(int(year), list(calendar.month_abbr).index(month), 1)
    month_end_date = datetime(int(year), list(calendar.month_abbr).index(month), selected_month_len[1])
    
    # calculates cohort rfm data for given month
    df_temp = cohort_rfm(df_data, month_end_date)
    
    retention_tab, cohort_tab = st.tabs(['Customer Filter', 'Cohort Analysis'])
    
    with retention_tab:
        # filters
        with st.expander('Customer Data Filters'):
            st.subheader('Last Transaction Date')
            min_txn, max_txn = st.columns(2)
            with min_txn:
                min_txn_date = st.date_input('Min Last Appointment Date:',
                          min_value = df_data.appointment_date.min(),
                          max_value = datetime.today(),
                          value = df_data.appointment_date.min())
                min_txn_date = min_txn_date.strftime('%Y/%m/%d')
            with max_txn:
                max_txn_date = st.date_input('Max Last Appointment Date:',
                          min_value = df_data.appointment_date.min(),
                          max_value = month_end_date,
                          value = month_start_date - relativedelta(months = 6))
                max_txn_date = max_txn_date.strftime('%Y/%m/%d')
        
            #df_retention = df_temp[(df_temp.last_txn_date >= min_txn_date) & (df_temp.last_txn_date <= max_txn_date)]
            df_retention = df_temp[df_temp.last_txn_date.between(min_txn_date, max_txn_date)].sort_values('month_diff',
                                                                                                      ascending = True)
            # df_retention = df_temp[df_temp.month_diff >= 6].sort_values(by = 'month_diff', 
            #                                                                   ascending = True)
        # customer table
        customer_retention_list = customer_search(df_data, df_retention)
        
        # master list
        df_merged = combine_customer_data(df_data, df_retention)
        
        month_year = '-'.join([month, year])
        stored_url = get_url(month_year)
        
        if stored_url is None:
            url = st.text_input('Google Sheet link to retention data')
            temp_button = st.button(f'Add Google Sheet URL for {month_year}')
            if temp_button:
                add_url(month_year, url)
                st.experimental_rerun()
            else:
                st.stop()
            
        else:
            gsheet = read_gsheet(stored_url, 'Masterlist')
            if gsheet is None:
                st.warning('Not able to find retention sheets.')
                write_button = st.button('Write retention data to google sheet?')
                if write_button:
                    write_retention_data(df_merged, stored_url)
                    st.experimental_rerun()
                else:
                    st.stop()
            else:
                # evals
                st.header('RETENTION TRACKING')
                show_retention_data(stored_url, month_start_date, month_end_date)
                
    with cohort_tab:
        
        st.header('Cohort Retention Analysis')
        # plot cohort_retention_chart
        st.write('''This chart shows the retention rate for customers of various cohorts
                 (grouped by first month of transaction). The data shows the portion
                 of customers that return with respect to the starting month.
                 ''')
        #cohort_pivot = cohort_analysis(df_data)
        cat_col, val_col, date_col = st.columns(3)
        with cat_col:
            cat = st.selectbox('Category',
                               options=['first_acq', 'model_year', 'brand',
                                        'service_category', 'mechanic_name', 'lead_source'],
                               index = 0,
                               help = 'Category to filter.')
        with val_col:
            opts = {'first_acq': ['None'],
                    'model_year': list(sorted(df_data[df_data['model_year'].notna()]['model_year'].unique())),
                    'brand' : list(sorted(df_data['brand'].unique())),
                    'service_category': list(np.unique(list(service_category_dict.keys()))),
                    'mechanic_name': sorted(df_data[df_data['mechanic_name'] != '']['mechanic_name'].unique()),
                    'lead_source': ['website_website', 'backend_website', 'website_backend']}
            
            val = st.selectbox('Category value',
                               options = opts[cat],
                               index = 0,
                               help = 'Value of category to filter.')
        with date_col:
            start_month = st.date_input('Starting month',
                          min_value = df_data.date.min().date(),
                          value = pd.to_datetime('2022-01-01'),
                          help = 'Starting month of data shown.')
        
        rate_or_actual = st.radio("Format",
                 options=["Rate", "Actual"],
                 index = 0,
                 help = 'Format of retention numbers shown.')
        # how to filter new and non-new customers
        plot_cohort_analysis(df_data, cat, val, start_month, rate_or_actual)
    
    
        
