import os
import numpy as np
import tensorflow as tf
import time

print("\n=============================================")
print("      SALIN SYSTEM DIAGNOSTIC TEST")
print("=============================================\n")

# --- TEST 1: CV MODEL LOADING ---
print("[1/2] Testing CV Model (The Eyes)...")
if os.path.exists('assets/mp_model_03.keras'):
    try:
        model = tf.keras.models.load_model('assets/mp_model_03.keras')
        print("   ✅ SUCCESS: 'mp_model_03.keras' loaded successfully.")
        
        # Verify Input Shape
        input_shape = model.input_shape
        print(f"   ℹ️  Model Input Shape: {input_shape} (Expected: (None, 120, 126))")
    except Exception as e:
        print(f"   ❌ FAILED: Model file exists but crashed on load.\n   Error: {e}")
else:
    print("   ❌ FAILED: 'assets/mp_model_03.keras' not found.")

# --- TEST 2: TRANSLATION LOGIC (The Brain) ---
print("\n[2/2] Testing Translation Logic (Rule-Based)...")

def evaluate_sentence_rules(gloss_text):
    """
    The exact logic we will use in app.py
    """
    if not gloss_text: return ""
    words = gloss_text.lower().split()
    
    # Rule A: Greetings
    if "hello" in words:
        return " ".join(w.capitalize() for w in words)
        
    # Rule B: Time Lists
    if len(words) > 1:
        formatted = [w.capitalize() for w in words]
        if len(words) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
            
    # Rule C: Single Word
    return words[0].capitalize()

# Define Test Cases
test_cases = [
    ("monday", "Monday"),
    ("monday friday", "Monday and Friday"),
    ("monday tuesday wednesday", "Monday, Tuesday, and Wednesday"),
    ("hello", "Hello"),
    ("hello monday", "Hello Monday"),
    ("hello today monday", "Hello today Monday"),
    ("hel", "Hel")
]

all_passed = True
print(f"   {'INPUT GLOSS':<30} | {'EXPECTED':<30} | {'ACTUAL':<30} | STATUS")
print("   " + "-"*110)

for inp, expected in test_cases:
    actual = evaluate_sentence_rules(inp)
    status = "✅ PASS" if actual == expected else "❌ FAIL"
    if actual != expected: all_passed = False
    print(f"   {inp:<30} | {expected:<30} | {actual:<30} | {status}")

print("\n=============================================")
if all_passed:
    print("✅ SYSTEM READY: Your logic is perfect. You can use app.py now.")
else:
    print("❌ SYSTEM ISSUE: Some logic tests failed.")
print("=============================================\n")