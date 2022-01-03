def chosung_tokenizer(string):
    CHOSUNGS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ',
                'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
                'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ',
                'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # JOONGSUNGS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ',
    #             'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    #             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
    #             'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
    #             'ㅣ']

    # JONGSUNGS = ['*', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ',
    #             'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    #             'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ',
    #             'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    #             'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ',
    #             'ㅌ', 'ㅍ', 'ㅎ']

    N_CHOSUNGS = 19
    N_JOONGSUNGS = 21
    N_JONGSUNGS = 28

    FIRST_HANGUL = 0xAC00 #'가'
    LAST_HANGUL = 0xD7A3 #'힣'

    result = []
    for char in string:
        check = ord(char)
        if not FIRST_HANGUL <= check <= LAST_HANGUL:
            result.append(char) # 한글 아니면 걍 붙인다.
        else:
            code = check - FIRST_HANGUL
            # jongsung_index = code % N_JONGSUNGS
            code //= N_JONGSUNGS
            # joongsung_index = code % N_JOONGSUNGS
            code //= N_JOONGSUNGS
            chosung_index = code

            result.append(CHOSUNGS[chosung_index])
            # result.append(JOONGSUNGS[joongsung_index])
            # result.append(JONGSUNGS[jongsung_index])

    return ''.join(result)

if __name__ == "__main__":
    print(chosung_tokenizer('나는 어제 치킨을 먹었다'))