# Pi Key -> DNA. Ini hukum alam, bukan coding.

DNA_MAP = {
    '00': 'A', '01': 'T', '10': 'C', '11': 'G'
}
REVERSE_DNA_MAP = {v: k for k, v in DNA_MAP.items()}

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_dna(binary):
    dna = ""
    for i in range(0, len(binary), 2):
        dna += DNA_MAP[binary[i:i+2]]
    return dna

def dna_to_binary(dna):
    binary = ""
    for base in dna:
        binary += REVERSE_DNA_MAP[base]
    return binary

def binary_to_text(binary):
    text = ""
    for i in range(0, len(binary), 8):
        text += chr(int(binary[i:i+8], 2))
    return text

def encode_pi_to_dna(pi_private_key):
    print(f"[ENCODER] Original Pi Key: {pi_private_key[:20]}...")
    binary = text_to_binary(pi_private_key)
    dna = binary_to_dna(binary)
    print(f"[ENCODER] DNA Sequence: {dna[:50]}... ({len(dna)} bases)")
    print(f"[ENCODER] Life Length: {len(dna)*0.34} nanometers")
    return dna

def decode_dna_to_pi(dna):
    binary = dna_to_binary(dna)
    pi_key = binary_to_text(binary)
    print(f"[DECODER] Resurrected Pi Key: {pi_key[:20]}...")
    return pi_key
