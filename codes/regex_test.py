import re

# Test cases
test_cases = [
    "High School Diploma",
    "high school diploma",
    "High School Degree",
    "high school degree",
    "High School",
    "GED",
    "cssd",
    "HSE",
    "Associate's Degree",
    "associate degree",
    "Associate's",
    "Bachelor of Arts",
    "Bachelor's Degree",
    "Bachelor's",
    "Bachelor",
    "College Degree",
    "College Diploma",
    "University Degree",
    "University Diploma",
    "Accredited College",
    "College",
    "Master's Degree",
    "Master of Science",
    "Master's",
    "Master",
    "Doctorate",
    "Ph.D.",
    "B.A.",
    "B.S.",
    "M.B.A.",
    "M.S.",
    "High School Diploma and GED",
    "Bachelor's degree in Science",
    "Master's of Arts",
    "HSE certificate",
    "B.S. in Biology",
    "M.A. in History",
    "Some random text B.A. not a degree",
    "This is a test for a bachelor's degree."
]

# Expected results for the test cases
expected_results = [
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True
]

# Original regex pattern (your provided one)
high_school = r'[Hh]igh [Ss]chool [Dd]iploma|[Hh]igh [Ss]chool [Dd]egree|[Hh]igh [Ss]chool|GED|CSSD|HSE|ged|cssd|hse|'
associate = r'[Aa]ssociate[\']?s [Dd]egree|[Aa]ssociate [Dd]egree|[Aa]ssociate[\']?s|'
bachelor = r'[Bb]achelor [Oo]f|[Bb]achelor[\']?s [Dd]egree|[Bb]achelor[\']?s|[Bb]achelor|[^a-zA-Z]B[.]?[AS][.]?[^a-zA-Z]|[Cc]ollege [Dd]egree|[Cc]ollege [Dd]iploma|[Uu]niversity [Dd]egree|[Uu]niversity [Dd]iploma|[Aa]ccredited [Cc]ollege|[Cc]ollege|[^a-zA-Z]B[.]?S[.]?N[.]?[^a-zA-Z]|'
master = r'[Mm]aster[\']?s [Dd]egree|[Mm]aster [Oo]f|[Mm]aster[\']?s|[Mm]aster|[^a-zA-Z]M[.]?B[.]?A[.]?[^a-zA-Z]|'
doctoral = r'[Dd]octorate|[Pp][Hh][.]?[Dd][.]?'
original_pattern = high_school + associate + bachelor + master + doctoral

# New regex pattern (the revised one)
new_pattern = (
    r"(?i)\b(?:high school diploma|high school degree|high school|ged|cssd|hse|"
    r"associate's degree|associate degree|associate's|"
    r"bachelor of|bachelor's degree|bachelor's|bachelor|"
    r"college degree|college diploma|university degree|university diploma|"
    r"accredited college|college|"
    r"master's degree|master of|master's|master|"
    r"doctorate|ph.d|"
    r"B\.?A\.?|B\.?S\.?|M\.?B\.?A\.?|M\.?S\.?)\b"
)

# Function to test both patterns
def test_patterns(test_cases):
    for case in test_cases:
        original_match = re.search(original_pattern, case) is not None
        new_match = re.search(new_pattern, case) is not None
        print(f"Testing: '{case}' => Original: {original_match}, New: {new_match}")

# Run the tests
test_patterns(test_cases)



draft a email saying  

I am not feeling well so I will taking day off. sorry to imform at the last minute 


