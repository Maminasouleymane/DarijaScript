from ply import yacc
import string
from lexer import tokens, lexer, is_ID

##########Pour personaliser les messages d'erreur du parser ###########
CBLUE = '\33[34m'
CRED = '\033[91m'
CEND = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
#########################################################################

##################################################################
################## Sym Table #####################################
##################################################################
CURR_SCOPE = 0


class SymTab:

    def __init__(self):

        self.scoped_symtab = [{}]

    def get_config(self):
        # nous faisons une copie superficielle de la table des symboles
        return list(self.scoped_symtab)

    def set_config(self, c):
        self.scoped_symtab = c

    def push_scope(self):
        # pousser un nouveau dictionnaire dans la pile - la pile s'agrandit vers la gauche
        self.scoped_symtab.insert(CURR_SCOPE, {})

    def pop_scope(self):
        # faire sortir le dictionnaire le plus à gauche de la pile
        if len(self.scoped_symtab) == 1:
            raise ValueError("cannot pop the global scope")
        else:
            self.scoped_symtab.pop(CURR_SCOPE)

    def declare_scalar(self, sym, init):
        # declare the scalar in the current scope: dict @ position 0

        # first we need to check whether the symbol was already declared
        # at this scope
        if sym in self.scoped_symtab[CURR_SCOPE]:
            raise ValueError("symbol {} already declared".format(sym))

        # enter the symbol in the current scope
        self.scoped_symtab[CURR_SCOPE].update({sym: ('scalar', init)})

    def declare_fun(self, sym, init):
        # declare a function in the current scope: dict @ position 0

        # first we need to check whether the symbol was already declared
        # at this scope
        if sym in self.scoped_symtab[CURR_SCOPE]:
            raise ValueError("symbol {} already declared".format(sym))

        # enter the function in the current scope
        self.scoped_symtab[CURR_SCOPE].update({sym: ('function', init)})

    def lookup_sym(self, sym):
        # find the first occurence of sym in the symtab stack
        # and return the associated value

        n_scopes = len(self.scoped_symtab)

        for scope in range(n_scopes):
            if sym in self.scoped_symtab[scope]:
                val = self.scoped_symtab[scope].get(sym)
                return val

        # not found
        raise ValueError("{} was not declared".format(sym))

    def update_sym(self, sym, val):
        # find the first occurence of sym in the symtab stack
        # and update the associated value

        n_scopes = len(self.scoped_symtab)

        for scope in range(n_scopes):
            if sym in self.scoped_symtab[scope]:
                scope_dict = self.scoped_symtab[scope]
                scope_dict[sym] = val
                return

        # not found
        raise ValueError("{} was not declared".format(sym))

###############################################################################
############################# STATE ##########################################
##############################################################################


class State:
    def __init__(self):
        self.initialize()

    def initialize(self):
        # symbol table to hold variable-value associations
        self.symbol_table = SymTab()

        # when done parsing this variable will hold our AST
        self.AST = None


state = State()


#########################################################################
############################ PARSER #####################################
#########################################################################
# définir la priorité et l'associativité
# NOTE: all operators need to have tokens
#       so that we can put them into the precedence table
precedence = (
    ('left', 'EQ', 'NE'),
    ('left', 'LT', 'BT'),
    ('left', 'LE', 'BE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'EUCLID'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'UMINUS')
)

#########################################################################
# règles de grammaire avec actions intégrées
#########################################################################


def p_prog(p):
    '''
    program : stmt_list
    '''
    state.AST = p[1]

#########################################################################


def p_stmt_list(p):
    '''
    stmt_list : stmt stmt_list
              | empty
    '''
    if (len(p) == 3):
        p[0] = ('seq', p[1], p[2])
    elif (len(p) == 2):
        p[0] = p[1]

#########################################################################


def p_stmt(p):
    '''
    stmt : sjel_3andk ID '(' opt_formal_args ')' stmt
         | sjel_3andk ID opt_init opt_semi
         | ID '=' exp opt_semi
         | khod ID opt_semi
         | byen_liya exp opt_semi
         | ID '(' opt_actual_args ')' opt_semi
         | raja3 opt_exp opt_semi
         | ma7ad '(' exp ')' stmt
         | i4a '(' exp ')' stmt opt_else
         | men '(' exp ')' hta exp b9adir stmt_list
         | kteb '(' opt_exp ')'
         | '{' stmt_list '}'

    '''
    if p[1] == 'sjel_3andk' and p[3] == '(':
        p[0] = ('fundecl', p[2], p[4], p[6])
    elif p[1] == 'sjel_3andk':
        p[0] = ('sjel_3andk', p[2], p[3])
    elif is_ID(p[1]) and p[2] == '=':
        p[0] = ('assign', p[1], p[3])
    elif p[1] == 'khod':
        p[0] = ('khod', p[2])
    elif p[1] == 'byen_liya':
        p[0] = ('byen_liya', p[2])
    elif is_ID(p[1]) and p[2] == '(':
        p[0] = ('callstmt', p[1], p[3])
    elif p[1] == 'raja3':
        p[0] = ('raja3', p[2])
    elif p[1] == 'ma7ad':
        p[0] = ('ma7ad', p[3], p[5])
    elif p[1] == 'i4a':
        p[0] = ('i4a', p[3], p[5], p[6])
    elif p[1] == 'men':
        p[0] = ('men', p[3], p[6], p[8])
    elif p[1] == 'kteb':
        p[0] = ('kteb', p[3])
    elif p[1] == '{':
        p[0] = ('block', p[2])
    else:
        raise ValueError("unexpected symbol {}".format(p[1]))

#########################################################################


def p_opt_formal_args(p):
    '''
    opt_formal_args : formal_args
                    | empty
    '''
    p[0] = p[1]

#########################################################################


def p_formal_args(p):
    '''
    formal_args : ID ',' formal_args
                | ID
    '''
    if (len(p) == 4):
        p[0] = ('seq', ('id', p[1]), p[3])
    else:
        p[0] = ('seq', ('id', p[1]), ('nil',))

#########################################################################


def p_opt_init(p):
    '''
    opt_init : '=' exp
             | empty
    '''
    if p[1] == '=':
        p[0] = p[2]
    else:
        p[0] = p[1]

#########################################################################


def p_opt_actual_args(p):
    '''
    opt_actual_args : actual_args
                    | empty
    '''
    p[0] = p[1]

#########################################################################


def p_actual_args(p):
    '''
    actual_args : exp ',' actual_args
                | exp
    '''
    if (len(p) == 4):
        p[0] = ('seq', p[1], p[3])
    else:
        p[0] = ('seq', p[1], ('nil',))

#########################################################################


def p_opt_exp(p):
    '''
    opt_exp : exp
            | empty
    '''
    p[0] = p[1]

#########################################################################


def p_opt_else(p):
    '''
    opt_else : mn_ghir stmt
             | empty
    '''
    if p[1] == 'mn_ghir':
        p[0] = p[2]
    else:
        p[0] = p[1]

#########################################################################


def p_binop_exp(p):
    '''
    exp : exp PLUS exp
        | exp MINUS exp
        | exp TIMES exp
        | exp DIVIDE exp
        | exp EUCLID exp
        | exp EQ exp
        | exp LE exp
        | exp NE exp
        | exp BE exp
        | exp BT exp
        | exp LT exp
        | exp w exp
        | exp wla exp
    '''
    p[0] = (p[2], p[1], p[3])


#########################################################################
def p_string_exp(p):
    '''
    exp : STRING
    '''
    p[0] = ('string', p[1])


#########################################################################
def p_double_exp(p):
    '''
    exp : DOUBLE
    '''
    p[0] = ('double', float(p[1]))

#########################################################################


def p_integer_exp(p):
    '''
    exp : INTEGER
    '''
    p[0] = ('integer', int(p[1]))

#########################################################################


def p_id_exp(p):
    '''
    exp : ID
    '''
    p[0] = ('id', p[1])


#########################################################################
def p_call_exp(p):
    '''
    exp : ID '(' opt_actual_args ')'
    '''
    p[0] = ('callexp', p[1], p[3])

#########################################################################


def p_paren_exp(p):
    '''
    exp : '(' exp ')'
    '''
    p[0] = ('paren', p[2])

#########################################################################


def p_uminus_exp(p):
    '''
    exp : MINUS exp %prec UMINUS
    '''
    p[0] = ('uminus', p[2])


#########################################################################
def p_opt_semi(p):
    '''
    opt_semi : ';'
             | empty
    '''
    pass

#########################################################################


def p_empty(p):
    '''
    empty :
    '''
    p[0] = ('nil',)

#########################################################################


def p_error(p):
    if p == None:
        token = "end of file"
    else:
        token = f"({p.value}) "

    print(f" {CRED}  {BOLD}  3andak mouchkil v lparser {CEND}: na9sak chi 7aja 9bel wla mn wra  {CBLUE} {token} {CEND}")


#########################################################################
# construction du  parser
#########################################################################
parser = yacc.yacc(debug=False, tabmodule='DARIJASCRIPTparsetab')


###########################################################################
###################### INTERPRETEUR #######################################
###########################################################################

# A tree walker to interpret DarijaScript programs

#########################################################################
# Utilisez le mécanisme d'exception pour renvoyer les valeurs des appels de fonction

class ReturnValue(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return(repr(self.value))

#########################################################################


def len_seq(seq_list):

    if seq_list[0] == 'nil':
        return 0

    elif seq_list[0] == 'seq':
        # unpack the seq node
        (SEQ, p1, p2) = seq_list

        return 1 + len_seq(p2)

    else:
        raise ValueError("unknown node type: {}".format(seq_list[0]))

#########################################################################


def eval_actual_args(args):

    if args[0] == 'nil':
        return ('nil',)

    elif args[0] == 'seq':
        # unpack the seq node
        (SEQ, p1, p2) = args

        val = walk(p1)

        return ('seq', val, eval_actual_args(p2))

    else:
        raise ValueError("unknown node type: {}".format(args[0]))

#########################################################################


def declare_formal_args(formal_args, actual_val_args):

    if len_seq(actual_val_args) != len_seq(formal_args):
        raise ValueError("actual and formal argument lists do not match")

    if formal_args[0] == 'nil':
        return

    # unpack the args
    (SEQ, (ID, sym), p1) = formal_args
    (SEQ, val, p2) = actual_val_args

    # declare the variable
    state.symbol_table.declare_scalar(sym, val)

    declare_formal_args(p1, p2)

#########################################################################


def handle_call(name, actual_arglist):

    (type, val) = state.symbol_table.lookup_sym(name)

    if type != 'function':
        raise ValueError("{} is not a function".format(name))

    # unpack the funval tuple
    (FUNVAL, formal_arglist, body, context) = val

    if len_seq(formal_arglist) != len_seq(actual_arglist):
        raise ValueError("function {} expects {} arguments".format(
            context, len_seq(formal_arglist)))

    # configurer l'environnement pour la portée statique puis exécuter la fonction
    # evaluate actuals in current symtab
    actual_val_args = eval_actual_args(actual_arglist)
    save_symtab = state.symbol_table.get_config()        # save current symtab
    # make function context current symtab
    state.symbol_table.set_config(context)
    state.symbol_table.push_scope()                      # push new function scope
    # declare formals in function scope
    declare_formal_args(formal_arglist, actual_val_args)

    return_value = None
    try:
        walk(body)                                       # exécuter la fonction
    except ReturnValue as val:
        return_value = val.value

    # NOTE: popping the function scope is not necessary because we
    # are restoring the original symtab configuration
    # restore original symtab config
    state.symbol_table.set_config(save_symtab)

    return return_value

#########################################################################
# node functions
#########################################################################


def seq(node):

    (SEQ, stmt, stmt_list) = node

    walk(stmt)
    walk(stmt_list)

#########################################################################


def nil(node):

    (NIL,) = node

    # do nothing!
    pass

#########################################################################


def fundecl_stmt(node):

    try:  # try the fundecl pattern without arglist
        (FUNDECL, name, (NIL,), body) = node

    except ValueError:  # try fundecl with arglist
        (FUNDECL, name, arglist, body) = node

        context = state.symbol_table.get_config()
        funval = ('funval', arglist, body, context)
        state.symbol_table.declare_fun(name, funval)

    else:  # fundecl pattern matched
        # no arglist is present
        context = state.symbol_table.get_config()
        funval = ('funval', ('nil',), body, context)
        state.symbol_table.declare_fun(name, funval)


#########################################################################
def sjel_3andk_stmt(node):

    try:  # try the declare pattern without initializer
        (sjel_3andk, name, (NIL,)) = node

    except ValueError:  # try declare with initializer
        (sjel_3andk, name, init_val) = node

        value = walk(init_val)
        state.symbol_table.declare_scalar(name, value)

    else:  # declare pattern matched
        # when no initializer is present we init with the value 0
        state.symbol_table.declare_scalar(name, 0)

#########################################################################


def assign_stmt(node):

    (ASSIGN, name, exp) = node

    value = walk(exp)
    state.symbol_table.update_sym(name, ('scalar', value))

#########################################################################


def khod_stmt(node):

    (khod, name) = node

    s = input("ch7al l9ima dyal " + name + " ? ")

    try:
        value = int(s)
    except ValueError:
        try:
            value = str(s)
        except ValueError:
            raise ValueError(
                "expected an integer or string or float  value for " + name)

    state.symbol_table.update_sym(name, ('scalar', value))

#########################################################################


def byen_liya_stmt(node):

    (PUT, exp) = node

    value = walk(exp)
    print("> {}".format(value))

#########################################################################


def call_stmt(node):

    (CALLSTMT, name, actual_args) = node

    handle_call(name, actual_args)

#########################################################################


def raja3_stmt(node):
    # si une valeur de retour existe le return stmt will l'enregistrera dans state object

    try:  # try return without exp
        (RETURN, (NIL,)) = node

    except ValueError:  # return with exp
        (RETURN, exp) = node

        value = walk(exp)
        raise ReturnValue(value)

    else:  # return without exp
        raise ReturnValue(None)

#########################################################################


def kteb_stmt(node):

    (kteb, exp) = node

    value = walk(exp)
    print("> {}".format(value))


#########################################################################
def ma7ad_stmt(node):

    (ma7ad, cond, body) = node

    value = walk(cond)
    while value != 0:

        walk(body)
        value = walk(cond)

########################################################################


def for_stmt(node):

    (FOR, var_ini, var_fin, body) = node

    value = walk(var_ini)
    val_fin = walk(var_fin)

    while value < val_fin:
        walk(body)
        value += 1

#########################################################################


def i4a_stmt(node):

    try:  # try the if-then pattern
        (i4a, cond, then_stmt, (NIL,)) = node

    except ValueError:  # if-then pattern didn't match
        (IF, cond, then_stmt, else_stmt) = node

        value = walk(cond)

        if value != 0:
            walk(then_stmt)
        else:
            walk(else_stmt)

    else:  # if-then pattern matched
        value = walk(cond)
        if value != 0:
            walk(then_stmt)

#########################################################################


def block_stmt(node):

    (BLOCK, stmt_list) = node

    state.symbol_table.push_scope()
    walk(stmt_list)
    state.symbol_table.pop_scope()
###########################################################################


def and_exp(node):
    (w, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)
    return 1 if v1 and v2 else 0

##############################################################


def or_exp(node):
    (wla, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 or v2 else 0

#########################################################################


def plus_exp(node):

    (PLUS, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)
    if isinstance(v1, int) and isinstance(v2, str):
        v_1 = str(v1)
        return v_1 + v2
    elif isinstance(v2, int) and isinstance(v1, str):
        v_2 = str(v2)
        return v1 + v_2
    else:
        return v1 + v2

#########################################################################


def minus_exp(node):

    (MINUS, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return v1 - v2

#########################################################################


def times_exp(node):

    (TIMES, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return v1 * v2

#########################################################################


def divide_exp(node):

    (DIVIDE, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return v1 // v2
#########################################################################


def euclid_exp(node):

    (EUCLID, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 % v2 else 0

#########################################################################


def eq_exp(node):

    (EQ, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 == v2 else 0

#########################################################################


def ne_exp(node):

    (NE, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 != v2 else 0
#########################################################################


def le_exp(node):

    (LE, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 <= v2 else 0

#########################################################################


def be_exp(node):

    (BE, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 >= v2 else 0

#########################################################################


def bt_exp(node):

    (BT, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 > v2 else 0

#########################################################################


def lt_exp(node):

    (LT, c1, c2) = node

    v1 = walk(c1)
    v2 = walk(c2)

    return 1 if v1 < v2 else 0

#########################################################################


def string_exp(node):

    (STRING, value) = node

# pour se deperaser de '' dans les chaines de caracteres

    new_string = str(value)
    return new_string.replace("'", " ")
#########################################################################


def integer_exp(node):

    (INTEGER, value) = node

    return value

#########################################################################


def double_exp(node):

    (INTEGER, value) = node

    return value
#########################################################################


def id_exp(node):

    (ID, name) = node

    (type, val) = state.symbol_table.lookup_sym(name)

    if type != 'scalar':
        raise ValueError("{} is not a scalar".format(name))

    return val

#########################################################################


def call_exp(node):
    # call_exp fonctionne comme call_stmt à l'exception  que nous devons renvoyer une valeur de retour

    (CALLEXP, name, args) = node

    return_value = handle_call(name, args)

    if return_value is None:
        raise ValueError("No return value from function {}".format(name))

    return return_value

#########################################################################


def uminus_exp(node):

    (UMINUS, exp) = node

    val = walk(exp)
    return - val


#########################################################################
def paren_exp(node):

    (PAREN, exp) = node

    # return the value of the parenthesized expression
    return walk(exp)

#########################################################################
# walk
#########################################################################


def walk(node):
    # node format: (TYPE, [child1[, child2[, ...]]])
    type = node[0]

    if type in dispatch_dict:
        node_function = dispatch_dict[type]
        return node_function(node)
    else:
        raise ValueError("walk: unknown tree node type: " + type)


# un dictionnaire pour associer les noeuds de l'arbre avec les functions
dispatch_dict = {
    'seq': seq,
    'nil': nil,
    'fundecl': fundecl_stmt,
    'sjel_3andk': sjel_3andk_stmt,
    'assign': assign_stmt,
    'khod': khod_stmt,
    'byen_liya': byen_liya_stmt,
    'callstmt': call_stmt,
    'raja3': raja3_stmt,
    'ma7ad': ma7ad_stmt,
    'i4a': i4a_stmt,
    'kteb': kteb_stmt,
    'men': for_stmt,
    'block': block_stmt,
    'string': string_exp,
    'integer': integer_exp,
    'double': double_exp,
    'id': id_exp,
    'callexp': call_exp,
    'paren': paren_exp,
    '+': plus_exp,
    '-': minus_exp,
    '*': times_exp,
    '/': divide_exp,
    '%': euclid_exp,
    '==': eq_exp,
    '<=': le_exp,
    '>=': be_exp,
    '>': bt_exp,
    '<': lt_exp,
    'ne': ne_exp,
    'and': and_exp,
    'or': or_exp,
    'uminus': uminus_exp
}

###################################################################################################################


def interp(input_stream):
    # initialisation de  state object
    state.initialize()

    # build the AST
    parser.parse(input_stream, lexer=lexer)

    # walk the AST
    walk(state.AST)


if __name__ == "__main__":
    #
    input_stream = \
        '''
        //da5al lcode dyal hna 
        kteb('salam world')
        '''

    # execute interpreter
    interp(input_stream=input_stream)
