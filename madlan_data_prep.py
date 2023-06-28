# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 00:34:57 2023

@author: user
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, date

path = "C:\\Users\\user\\Desktop\\נטע\\לימודים\\שנה ג\\סמסטר ב\\ניתוח נתונים מתקדם\\"
filename = "Train_Set.xlsx"
data = path + filename
df = pd.read_excel(data)

#טיפול בעמודות של מחיר ושטח על מנת שנוכל להמיר את העמודות לערך מספרי 
def clean_price_or_area(value):
    if pd.isnull(value) or value == '':
        return np.nan
    elif value == "₪3,950,000TOP10 במדד המתווכים":
        value = value.replace("₪", "").replace(",","").replace("TOP10","").replace(" במדד המתווכים", "")
    else:
        value = ''.join(filter(str.isdigit, str(value)))
    if value == '':
        return np.nan
    return value


#הורדת סימני פיסוק מהעמודות הרלוונטיות 
def clean_punctuation(value):
    if pd.isnull(value) or value == '':
        return None
    if isinstance(value, str):
        punctuation =".,;:><|-_\/?*&^)(#!+="
        translator = str.maketrans('', '', punctuation)
        value = value.translate(translator)
    return value


##הוספה של עמודת קומה בלבד 
def add_floor_column(val):
    if isinstance(val, (int, float)):
        val = str(val)
    if 'קרקע' in val:
        val = 0
    elif 'מרתף' in val:
        val = -1
    else:
        match = re.search(r'\d+', val)
        if match:
            val = int(match.group())
        else:
            val = None
    return val


#הוספה של עמודת סהכ קומות בביניין 
def add_total_floor_column(val):
    if isinstance(val, (int, float)):
        val = str(val)
    match = re.findall(r'\d+', val)
    if match:
        val = float(match[-1])
    else:
        val = None
    return val


#שינוי תאריך הכניסה לערכים קטגוריאלים
def transform_entrance_date(item):
    if isinstance(item, str):
        if item == 'מיידי':
            return 'less_than_6 months'
        elif item == 'גמיש':
            return 'flexible'
        elif item == 'לא צויין':
            return 'not_defined'
        else:
            try:
                specific_date = datetime.strptime(item, '%d/%m/%Y').date()
            except ValueError:
                return 'unknown'
    elif isinstance(item, datetime):
        specific_date = item.date()
    else:
        return 'unknown'

    today = date.today()
    months_diff = (specific_date.year - today.year) * 12 + (specific_date.month - today.month)

    if months_diff <= 6:
        return 'less_than_6 months'
    elif months_diff > 6 and months_diff <= 12:
        return 'months_6_12'
    else:
        return 'above_year'
    return item
  
  
#המרת עמודות בינאריות ל-0 ו-1
def convert_to_binary(val):
    true_values = ['true', 'yes', "כן", "יש","1"]
    false_values = ['false', 'no', "לא","אין","0","nan"]
    val = str(val)
    val = val.lower()
    for t in true_values:
        if t in val:
            return 1
    for f in false_values:
        if f in val:
            return 0
    if val.startswith('נגיש'):
        return 1
    return val


#טיפול בעמודה של מספר חדרים כי זה נכנס למודל 
def clean_room_number(value):
    pattern = r"[^\d.]"
    cleaned_value = re.sub(pattern, "", str(value))
    if cleaned_value.strip() == '':
        return np.nan
    else:
        return float(cleaned_value)


# פה ניתן להכניס את השם של הדאטה וזה יבצע טיפול בדאטה
def prepare_data(df):
    df = df.copy()
    df['price'] = df['price'].apply(clean_price_or_area)
    df = df.dropna(subset=['price'])
    df['price'] = df['price'].astype(float)
    
    df['Area'] = df['Area'].apply(clean_price_or_area)
    df['Area']= df['Area'].astype(float)
    
    df['city_area'] = df['city_area'].apply(clean_punctuation)
    df['description '] = df['description '].apply(clean_punctuation)
    df['Street'] = df['Street'].apply(clean_punctuation)
    
    df['floor'] = df['floor_out_of'].apply(add_floor_column)
    
    df['total_floors'] = df['floor_out_of'].apply(add_total_floor_column)
    
    df['entrance_date'] = [transform_entrance_date(item) for item in df['entranceDate ']]
    
    df['hasElevator '] = df['hasElevator '].apply(convert_to_binary)
    df['hasParking '] = df['hasParking '].apply(convert_to_binary)
    df['hasBars '] = df['hasBars '].apply(convert_to_binary)
    df['hasStorage '] = df['hasStorage '].apply(convert_to_binary)
    df['hasAirCondition '] = df['hasAirCondition '].apply(convert_to_binary)
    df['hasBalcony '] = df['hasBalcony '].apply(convert_to_binary)
    df['hasMamad '] = df['hasMamad '].apply(convert_to_binary)
    df['handicapFriendly '] = df['handicapFriendly '].apply(convert_to_binary)
    
    df["room_number"] = df["room_number"].apply(clean_room_number)
    
    features = ["City","type","Area","hasElevator ","hasParking ","hasBalcony ","hasMamad ","entrance_date","price"]
    
    return df[features]