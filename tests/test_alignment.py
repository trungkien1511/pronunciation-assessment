"""Test grapheme-phoneme alignment với 45 từ phổ biến."""

# Copy logic trực tiếp, không import module (tránh load G2P chậm)
CHAR_TO_PHONEMES = {
    'a': {'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'EY'},
    'b': {'B'}, 'c': {'K', 'S', 'CH', 'SH'}, 'd': {'D', 'JH'},
    'e': {'AH', 'EH', 'ER', 'EY', 'IH', 'IY'},
    'f': {'F'}, 'g': {'G', 'JH'}, 'h': {'HH'},
    'i': {'AH', 'AY', 'IH', 'IY'},
    'j': {'JH', 'Y'}, 'k': {'K'},
    'l': {'L', 'AH'}, 'm': {'M'}, 'n': {'N', 'NG', 'AH'},
    'o': {'AA', 'AH', 'AO', 'AW', 'OW', 'OY', 'UH', 'UW'},
    'p': {'P'}, 'q': {'K'}, 'r': {'R', 'ER', 'AH'},
    's': {'S', 'SH', 'Z', 'ZH'}, 't': {'T', 'CH', 'SH'},
    'u': {'AH', 'UH', 'UW', 'W', 'Y', 'ER'},
    'v': {'V'}, 'w': {'W'}, 'x': {'K', 'S', 'Z'},
    'y': {'AY', 'IH', 'IY', 'Y'}, 'z': {'S', 'Z', 'ZH'},
}
DIGRAPH_TO_PHONEMES = {
    'ch': {'CH', 'K', 'SH'}, 'ck': {'K'}, 'dg': {'JH'},
    'gh': {'F', 'G'}, 'gn': {'N'}, 'kn': {'N'}, 'ng': {'NG'},
    'ph': {'F'}, 'sh': {'SH'}, 'th': {'DH', 'TH'},
    'wh': {'HH', 'W'}, 'wr': {'R'},
    'bb': {'B'}, 'cc': {'K', 'S'}, 'dd': {'D'}, 'ff': {'F'},
    'gg': {'G', 'JH'}, 'll': {'L'}, 'mm': {'M'}, 'nn': {'N'},
    'pp': {'P'}, 'rr': {'R'}, 'ss': {'S', 'Z'}, 'tt': {'T'}, 'zz': {'Z'},
}

def align_chars_phonemes(word, phonemes):
    w = word.lower()
    result = []
    ci = 0
    pi = 0
    while pi < len(phonemes) and ci < len(w):
        ph = phonemes[pi]
        matched = False
        if ci + 1 < len(w):
            di = w[ci:ci + 2]
            if di in DIGRAPH_TO_PHONEMES and ph in DIGRAPH_TO_PHONEMES[di]:
                result.append((word[ci:ci + 2], ph))
                ci += 2; pi += 1; matched = True
        if not matched:
            ch = w[ci]
            if ch in CHAR_TO_PHONEMES and ph in CHAR_TO_PHONEMES[ch]:
                result.append((word[ci:ci + 1], ph))
                ci += 1; pi += 1; matched = True
        if not matched:
            if result:
                prev_chars, prev_ph = result[-1]
                result[-1] = (prev_chars + word[ci], prev_ph)
            else:
                result.append((word[ci], None))
            ci += 1
    while ci < len(w):
        if result:
            prev_chars, prev_ph = result[-1]
            result[-1] = (prev_chars + word[ci], prev_ph)
        ci += 1
    while pi < len(phonemes):
        result.append(("", phonemes[pi]))
        pi += 1
    merged = []; buf = ""
    for chars, ph in result:
        if ph is None: buf += chars
        else: merged.append((buf + chars, ph)); buf = ""
    if buf and merged:
        c, p = merged[-1]; merged[-1] = (c + buf, p)
    elif buf:
        merged.append((buf, ""))
    return merged

# ===================== TEST CASES =====================
test_cases = [
    ('apple',     ['AE','P','AH','L']),
    ('bottle',    ['B','AA','T','AH','L']),
    ('little',    ['L','IH','T','AH','L']),
    ('button',    ['B','AH','T','AH','N']),
    ('kitten',    ['K','IH','T','AH','N']),
    ('happy',     ['HH','AE','P','IY']),
    ('letter',    ['L','EH','T','ER']),
    ('butter',    ['B','AH','T','ER']),
    ('puppy',     ['P','AH','P','IY']),
    ('rabbit',    ['R','AE','B','AH','T']),
    ('grape',     ['G','R','EY','P']),
    ('cake',      ['K','EY','K']),
    ('bike',      ['B','AY','K']),
    ('home',      ['HH','OW','M']),
    ('phone',     ['F','OW','N']),
    ('write',     ['R','AY','T']),
    ('knight',    ['N','AY','T']),
    ('know',      ['N','OW']),
    ('comb',      ['K','OW','M']),
    ('thumb',     ['TH','AH','M']),
    ('island',    ['AY','L','AH','N','D']),
    ('listen',    ['L','IH','S','AH','N']),
    ('school',    ['S','K','UW','L']),
    ('ship',      ['SH','IH','P']),
    ('think',     ['TH','IH','NG','K']),
    ('chat',      ['CH','AE','T']),
    ('whale',     ['W','EY','L']),
    ('through',   ['TH','R','UW']),
    ('thought',   ['TH','AO','T']),
    ('enough',    ['IH','N','AH','F']),
    ('night',     ['N','AY','T']),
    ('light',     ['L','AY','T']),
    ('people',    ['P','IY','P','AH','L']),
    ('table',     ['T','EY','B','AH','L']),
    ('purple',    ['P','ER','P','AH','L']),
    ('circle',    ['S','ER','K','AH','L']),
    ('orange',    ['AO','R','AH','N','JH']),
    ('chocolate', ['CH','AA','K','AH','L','AH','T']),
    ('elephant',  ['EH','L','AH','F','AH','N','T']),
    ('banana',    ['B','AH','N','AE','N','AH']),
    ('yellow',    ['Y','EH','L','OW']),
    ('hello',     ['HH','AH','L','OW']),
    ('world',     ['W','ER','L','D']),
    ('mother',    ['M','AH','DH','ER']),
    ('beautiful', ['B','Y','UW','T','AH','F','AH','L']),
]

bug_count = 0
for word, phs in test_cases:
    cmap = align_chars_phonemes(word, phs)
    has_empty = any(c == '' for c, p in cmap)
    chars_joined = ''.join(c for c, p in cmap)
    ok = chars_joined.lower() == word.lower()
    
    flag = ''
    if has_empty: flag += ' [EMPTY_CHAR]'
    if not ok: flag += f' [MISMATCH: got={chars_joined}]'
    
    cmap_str = '  '.join([f'{c}={p}' for c, p in cmap])
    status = 'OK ' if not flag else 'BUG'
    if flag: bug_count += 1
    print(f'{status}  {word:<12}  {cmap_str}{flag}')

print(f'\n=== {bug_count} BUG / {len(test_cases)} tu ===')
