import re
from ply import lex

##########Pour personaliser les messages d'erreur du lexer ###########
CBLUE   = '\33[34m'
CRED = '\033[91m'
CEND = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
#######################################################################

reserved = {
    'khod': 'khod',
    'byen_liya': 'byen_liya',
    'i4a': 'i4a',
    'mn_ghir': 'mn_ghir',
    'ma7ad': 'ma7ad',
    'men': 'men',
    'hta' : 'hta',
    'b9adir' : 'b9adir',
    'kteb'  : 'kteb',
    'w' : 'w',
    'wla'  : 'wla',
    'sjel_3andk': 'sjel_3andk',
    'raja3': 'raja3'
}

literals = [',', ';', '=', '(', ')', '{', '}']

tokens = [
             'PLUS', 'MINUS', 'TIMES', 'DIVIDE','EUCLID',
             'EQ', 'LE','NE','BE','BT','LT',
             'INTEGER', 'ID','DOUBLE','STRING'

         ] + list(reserved.values())

t_STRING = r'\'.*?\''
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_EUCLID = r'%'
t_EQ = r'=='
t_NE = r'!='
t_LE = r'<='
t_BE = r'>='
t_BT = r'>'
t_LT = r'<'
t_ignore = ' \t'


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Vérifiez les mots réservés
    return t

def t_DOUBLE(t):
    r'[0-9]+\.[0-9]+'
    try:
        t.value = float(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t

def t_INTEGER(t):
    r'[0-9]+'
    return t

def t_COMMENT(t):
    r'//.*'
    pass

def t_NEWLINE(t):
    r'\n'
    pass

def is_ID(s):
    m = re.match(r'[a-zA-Z_][a-zA-Z_0-9]*', s)

    if s in list(reserved.keys()):
        return False
    elif m and len(m.group(0)) == len(s):
        return True
    else:
        return False

def t_error(t):

    print(f"{CRED}  {BOLD}  3andak mouchkil v lparser {CEND}:  {CBLUE} '{t.value[0]}' {CEND} ma kaynach v DARIJASCRIPT ")
    t.lexer.skip(1)


# build the lexer
lexer = lex.lex(debug=0)