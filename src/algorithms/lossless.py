import heapq
import math
from collections import Counter
from .utils import build_frequency


# --- RLE ---
def run_length_encode(data: str) -> str:
    if not data:
        return ""
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}|{data[i - 1]}")
            count = 1
    encoded.append(f"{count}|{data[-1]}")
    return "".join(encoded)


def run_length_decode(encoded: str) -> str:
    decoded = []
    i = 0
    n = len(encoded)
    while i < n:
        count_str = ""
        while i < n and encoded[i] != "|":
            count_str += encoded[i]
            i += 1
        i += 1
        if i < n:
            char = encoded[i]
            if count_str.isdigit():
                decoded.append(char * int(count_str))
            i += 1
    return "".join(decoded)


# --- Huffman ---
def build_heap(freq):
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    return heap


def build_codes(heap):
    local_heap = heap[:]
    while len(local_heap) > 1:
        smallest = heapq.heappop(local_heap)
        secsmallest = heapq.heappop(local_heap)
        for pair in smallest[1:]:
            pair[1] = "0" + pair[1]
        for pair in secsmallest[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(
            local_heap, [smallest[0] + secsmallest[0]] + smallest[1:] + secsmallest[1:]
        )
    if not local_heap:
        return {}
    return dict(local_heap[0][1:])


def huffman_encode(text, codes):
    return "".join(codes[ch] for ch in text)


def huffman_decode(encoded_text, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_text = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text += reverse_codes[current_code]
            current_code = ""
    return decoded_text


# --- Golomb ---
def unary_encode(q: int) -> str:
    return "1" * q + "0"


def golomb_encode(n: int, m: int) -> str:
    q = n // m
    r = n % m
    quotient_code = unary_encode(q)
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        remainder_code = format(r, f"0{k}b")
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        if r < T:
            remainder_code = format(r, f"0{b-1}b")
        else:
            remainder_code = format(r + T, f"0{b}b")
    return quotient_code + remainder_code


def golomb_decode_stream(bitstream, m):
    decoded_text = ""
    idx = 0
    n_len = len(bitstream)
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        is_power_2 = True
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        is_power_2 = False

    while idx < n_len:
        q = 0
        while idx < n_len and bitstream[idx] == "1":
            q += 1
            idx += 1
        idx += 1
        r = 0
        if is_power_2:
            if idx + k > n_len:
                break
            r = int(bitstream[idx : idx + k], 2)
            idx += k
        else:
            if idx + (b - 1) > n_len:
                break
            temp_r = int(bitstream[idx : idx + (b - 1)], 2)
            if temp_r < T:
                r = temp_r
                idx += b - 1
            else:
                if idx + b > n_len:
                    break
                r = int(bitstream[idx : idx + b], 2) - T
                idx += b
        try:
            decoded_text += chr(q * m + r)
        except:
            decoded_text += "?"
    return decoded_text


def get_golomb_bits_len(val, m):
    q = val // m
    r = val % m
    len_unary = q + 1
    if (m & (m - 1)) == 0:
        k = int(math.log2(m))
        len_rem = k
    else:
        b = math.ceil(math.log2(m))
        T = 2**b - m
        len_rem = b - 1 if r < T else b
    return len_unary + len_rem


# --- LZW ---
def lzw_encode(text):
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    current_c = ""
    result = []
    for next_n in text:
        combined = current_c + next_n
        if combined in dictionary:
            current_c = combined
        else:
            result.append(dictionary[current_c])
            dictionary[combined] = next_code
            next_code += 1
            current_c = next_n
    if current_c != "":
        result.append(dictionary[current_c])
    return result, dictionary


def lzw_decode(codes):
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    if not codes:
        return ""
    prev_entry = dictionary[codes[0]]
    result = prev_entry
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = prev_entry + prev_entry[0]
        result += entry
        dictionary[next_code] = prev_entry + entry[0]
        next_code += 1
        prev_entry = entry
    return result
