import sys
import anyascii
from phase3_analyze_misses import clean_name, squish, clean_addr

def clean_name_v2(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = anyascii.anyascii(s).lower()
    import re
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s: s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie)\b', ' ', s)
    return ' '.join(s.split())

def clean_addr_v2(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = anyascii.anyascii(s).lower()
    import re
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl)\b', ' ', s)
    return ' '.join(s.split())

print("Testing clean_name_v2:")
print("Hindi:", repr(clean_name_v2('रेड वेंचर्स प्राइवेट लिमिटेड')))
print("Telugu:", repr(clean_name_v2('ఇంటర్నేషనల్ సిస్టమ్స్ ప్రైవేట్ లిమిటెడ్')))
print("Kannada:", repr(clean_name_v2('ಬಾಲಾಜಿ ಬಿಸಿನೆಸ್ ప్రైవేట్ లిమిటెడ్')))
print("English:", repr(clean_name_v2('Apple Corporation Inc.')))
