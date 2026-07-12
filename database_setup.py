# database_setup.py
import sqlite3

# Database se connect karein (Agar file nahi hai to ban jayegi)
conn = sqlite3.connect('plant_disease.db')
cursor = conn.cursor()

# Disease Information Table banayein
cursor.execute('''
CREATE TABLE IF NOT EXISTS disease_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT UNIQUE,
    
    name_en TEXT,
    remedy_en TEXT,
    
    name_ur TEXT,
    remedy_ur TEXT,
    
    name_pb TEXT,
    remedy_pb TEXT,
    
    name_sd TEXT,
    remedy_sd TEXT
)
''')

# Dummy Data Insert karte hain (Sample ke liye Tomato Leaf Mold)
# Jab aapka model train ho jaye, toh aap saari classes ka data isi tarah add kar sakte hain
sample_data = (
    "Tomato___Leaf_Mold",
    
    # English
    "Tomato Leaf Mold", 
    "Ensure good air circulation in the greenhouse. Use appropriate fungicides if the infection is severe.",
    
    # Urdu
    "ٹماٹر کے پتوں کا فنگس (Leaf Mold)", 
    "پودوں کے درمیان ہوا کا گزر بہتر بنائیں۔ اگر بیماری زیادہ ہو تو فنگسائڈ کا سپرے کریں۔",
    
    # Punjabi
    "ٹماٹر دے پتیاں دی فنگس", 
    "بوٹیاں دے وچ ہوا دا لنگھنا بہتر کرو۔ جے بیماری زیادہ ہووے تے دوائی دا سپرے کرو۔",
    
    # Sindhi
    "ٹماٹر جي پنن جي ڦڦندي (Leaf Mold)", 
    "پوکن جي وچ ۾ ہوا جو لنگھہ بہتر بڻايو. جيڪڏہن بيماري وڌيڪ ہوجي ته فنگسائيڊ جو اسپري ڪريو."
)

try:
    cursor.execute('''
    INSERT OR REPLACE INTO disease_info 
    (class_name, name_en, remedy_en, name_ur, remedy_ur, name_pb, remedy_pb, name_sd, remedy_sd)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_data)
    conn.commit()
    print("Database aur sample multilingual data successfully tayyar hai!")
except Exception as e:
    print(f"Error aaya: {e}")
finally:
    conn.close()