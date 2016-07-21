INTEGER, BOOL = 'INTEGER', 'BOOL'
PLUS, MINUS, MULTIPLY, EOF = 'PLUS', 'MINUS', 'MULTIPLY', 'EOF'
DIVIDE = 'DIVIDE'
LPAREN, RPAREN = 'LPAREN', 'RPAREN'
BEGIN, END = 'BEGIN', 'END'
ID, ASSIGN = 'ID', 'ASSIGN'
DOT, COMMA, SEMI = 'DOT', 'COMMA', 'SEMI'
PPLUS, MMINUS = 'PPLUS', 'MMINUS'
PEQUALS, MEQUALS = 'PEQUALS', 'MEQUALS'
WHILE, FOR, IF, ELSE, THEN, ENDIF = 'WHILE', 'FOR', 'IF', 'ELSE', 'THEN', 'ENDIF'
GTHAN,LTHAN,GTEQUALS,LTEQUALS = 'GTHAN', 'LTHAN', 'GTEQUALS', 'LTEQUALS'
EQUALS,NOT,NEQUALS = 'EQUALS', 'NOT', 'NEQUALS'
AND, OR = 'AND', 'OR'
DOTRANGE = 'DOTRANGE'
COMMENT, BLOCK = 'COMMENT', 'BLOCK'

import inspect
import unicodedata

class AST(object):
    pass

class BinOp(AST):
    def __init__(self, left, op, right, env=None):
        self.left = left
        self.token = self.op = op
        self.right = right

class Primitive(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
        
class UnaryOp(AST):
    def __init__(self, op, expr, env=None):
        self.token = self.op = op
        self.expr = expr
        
class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self, env):
        self.env = env
        self.children = []
        
class Conditional(AST):
    """Represents a conditional fork"""
    def __init__(self, pred, conseq, alt=None, env=None):
        self.pred = pred
        self.conseq = conseq
        self.alt = alt

class While(AST):
    def __init__(self, pred, conseq, env=None):
        self.pred = pred
        self.conseq = conseq

class For(AST):
    def __init__(self, var, assign, range, cond, post, conseq, env=None):
        self.assign = assign
        self.range = range
        self.cond = cond
        self.post = post
        self.conseq = conseq
        self.var = var

class Assign(AST):
    def __init__(self, left, op, right, env=None):
        self.left = left
        self.token = self.op = op
        self.right = right

class Var(AST):
    def __init__(self, token, env=None):
        self.env = env
        self.token = token
        self.value = token.value

class NoOp(AST):
    pass

class Env(object):
    def __init__(self, parent=None):
        self.vars = {} if parent is None else parent.vars.copy()
        self.parent = parent

    def extend(self):
        return Env(parent=self)

    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.vars:
                if scope.parent is not None and name not in scope.parent.vars:
                    return scope
            scope = scope.parent
        return None

    def set(self, name, value):
        scope = self.lookup(name)
        scope = self if scope is None else scope
        scope.vars[name] = value
        return value

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        else:
            return None

    def define(self, name, value):
        self.vars[name] = value
        return value

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value
        
    def __str__(self):
        return 'Token({type}, {value})'.format(
            type = self.type,
            value = repr(self.value)
        )
        
    def __repr(self):
        return self.__str__()
        
RESERVED_KEYWORDS = {
    'BEGIN': Token(BEGIN, 'BEGIN'),
    'begin': Token(BEGIN, 'begin'),
    'END': Token(END, 'END'),
    'end': Token(END, 'end'),
    'is': Token(ASSIGN, 'is'),
    'while': Token(WHILE, 'while'),
    'if': Token(IF, 'if'),
    'else': Token(ELSE, 'else'),
    'endif': Token(ENDIF, 'endif'),
    'then': Token(THEN, 'then'),
    'true': Token(BOOL, True),
    'false': Token(BOOL, False),
    'equals': Token(EQUALS, 'equals'),
    'not': Token(NOT, 'not'),
    'and': Token(AND, 'and'),
    'or': Token(OR, 'and'),
    'for': Token(FOR, 'for'),
}        
    
class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.prev_char = None
        self.col_num = 0
        self.line_num = 1
        
    def error(self):
        raise Exception('Invalid character: ' + repr(self.current_char))
        
    def advance(self, step=None):
        if step is None:
            step = 1
        self.pos += step
        self.col_num += step
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            
    def _id(self):
        result = ''
        while self.current_char is not None and self.is_valid_varchar(self.current_char):
            result += self.current_char
            self.advance()
            
        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        if token.type == NOT and self.peekToken().type == EQUALS:
            self.get_next_token()
            return Token('NEQUALS', 'not equals')
        return token
        
    def is_valid_varchar(self, char):
        if char.isalnum() or char in '_':
            return True
        return False
    
    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
            
    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)
        
    def get_next_token(self):

        while self.current_char is not None:
            if self.current_char.isspace():
                if self.current_char == '\n':
                    self.line_num += 1
                self.skip_whitespace()
                self.col_num = 1
                continue
            
            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())
                
            if self.current_char == '+' and self.peek() == '+' and self.peek(-1).isalnum():
                self.advance(2)
                return Token(PPLUS, '++')
                
            if self.current_char == '+' and self.peek() == '=':
                self.advance(2)
                return Token(PEQUALS, '+=')
                
            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')
                
            if self.current_char == '-' and self.peek() == '-' and self.peek(-1).isalnum():
                self.advance(2)
                return Token(MMINUS, '--')
                
            if self.current_char == '-' and self.peek() == '=':
                self.advance(2)
                return Token(MEQUALS, '-=')
                
            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')
                
            if self.current_char == '*':
                self.advance()
                return Token(MULTIPLY, '*')
            
            if self.current_char == '$' and self.peek() == '$':
                self.advance(2)
                return Token(BLOCK, '$$')

            if self.current_char == '$':
                self.advance()
                return Token(COMMENT, '$')

            if self.current_char == '/':
                self.advance()
                return Token(DIVIDE, '/')
            
            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')
            
            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')
            
            if self.current_char.isalpha():
                return self._id()
                
            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')
                
            if self.current_char == '.' and self.peek() == '.':
                self.advance(2)
                return Token(DOTRANGE, '..')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '>' and self.peek() == '=':
                self.advance(2)
                return Token(GTEQUALS, '>=')

            if self.current_char == '>':
                self.advance()
                return Token(GTHAN, '>')

            if self.current_char == '<' and self.peek() == '=':
                self.advance(2)
                return Token(LTEQUALS, '<=')

            if self.current_char == '<':
                self.advance()
                return Token(LTHAN, '<')
                
            self.error()
            
        return Token(EOF, None)
        
    def peek(self, step=None):
        if step is None:
            step = 1;
        peek_pos = self.pos+step
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]
            
    def peekToken(self, n=None):
        if n is None:
            n = 1
        if n == 0:
            return

        current_pos = self.pos
        current_char = self.current_char
        token = self.get_next_token()
        self.peekToken(n-1)
        self.pos = current_pos
        self.current_char = current_char
        return token

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.prev_token = None
        self.current_token = self.lexer.get_next_token()
        
    def error(self, expect=None):
        lnum = repr(self.lexer.line_num)
        cnum = repr(self.lexer.col_num)
        token_val = repr(self.current_token.value)
        exp_msg = '' if expect is None else ', Expected: ' + expect
        raise Exception('Invalid syntax on line '+lnum+', col '+cnum+': '+token_val+exp_msg)
        
    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.prev_token = self.current_token
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(token_type)

    def variable(self, env):
        node = Var(self.current_token, env)
        self.eat(ID)
        return node

    def assignment_statement(self, env):
        left = self.variable(env)
        self.eat(ASSIGN)
        if self.lexer.peekToken().type == ASSIGN:
            right = self.assignment_statement(env)
        else:
            right = self.expr_a(env)
        return Assign(left, Token(Assign, 'is'), right, env)

    def comment(self):
        self.eat(COMMENT)
        while self.lexer.current_char != '\n':
            self.lexer.get_next_token()
        self.prev_token = Token(COMMENT,'$')
        self.current_token = self.lexer.get_next_token()
        return self.empty()
        
    def block_comment(self):
        self.eat(BLOCK)
        while self.current_token.type != BLOCK:
            self.current_token = self.lexer.get_next_token()
        self.eat(BLOCK)
        return self.empty()

    def incdec_statement(self, env):
        """
        incdec is same as varASSIGN(varADD1)
        """
        left = self.variable(env)
        token = self.current_token
        one = Primitive(Token(INTEGER, 1))
        if token.type == PPLUS:
            self.eat(PPLUS)
            right = BinOp(left, Token(PLUS, '+'), one, env)
        else:
            self.eat(MMINUS)
            right = BinOp(left, Token(MINUS, '-'), one, env)
        node = Assign(left, token, right)
        return node
        
    def incdec_assign_statement(self, env):
        left = self.variable(env)
        token = self.current_token
        if token.type == PEQUALS:
            self.eat(PEQUALS)
            right = BinOp(left, Token(PLUS, '+'), self.expr_d(), env)
        elif token.type == MEQUALS:
            self.eat(MEQUALS)
            right = BinOp(left, Token(MINUS, '-'), self.expr_d(), env)
        node = Assign(left, token, right, env)
        return node

    def statement(self, env):
        if self.current_token.type == BEGIN:
            node = self.compound_statement(env)
        elif self.current_token.type == IF:
            in_else = self.prev_token.type == ELSE
            node = self.if_statement(env=env,in_else=in_else)
        elif self.current_token.type == WHILE:
            node = self.while_loop(env)
        elif self.current_token.type == FOR:
            node = self.for_loop(env)
        elif self.current_token.type == ID:
            next_token = self.lexer.peekToken()
            if next_token.type == ASSIGN:
                node = self.assignment_statement(env)
            elif next_token.type in (PPLUS, MMINUS):
                node = self.incdec_statement(env)
            elif next_token.type in (PEQUALS, MEQUALS):
                node = self.incdec_assign_statement(env)
            else:
                node = self.empty()
        else:
            node = self.empty()
        return node

    def statement_list(self, env):
        node = self.statement(env)

        results = [node]
        while self.current_token.type in (SEMI,COMMENT,BLOCK):
            if self.current_token.type == SEMI:
                self.eat(SEMI)
            elif self.current_token.type == COMMENT:
                self.comment()
            else:
                self.block_comment()
            results.append(self.statement(env))
            
        if self.current_token.type == ID:
            self.error()
            
        return results

    def while_loop(self, env):
        self.eat(WHILE)
        predicate = self.expr_a(env)
        self.eat(THEN)
        consequent = self.compound_statement(env)
        self.eat(END)
        return While(predicate, consequent, env)

    def for_loop(self, env):
        var = begin_range = end_range = cond = post = None
        self.eat(FOR)

        env = env.extend()

        assignment = self.assignment_statement(env)
        begin_range = None
        if self.current_token.type == DOTRANGE:
            var = assignment.left
            begin_range = assignment.right
            self.eat(DOTRANGE)
            end_range = self.expr_a(env)

            if self.current_token.type == COMMA:
                # if post step is specified
                self.eat(COMMA)
                post = self.statement(env)
            else:
                # otherwise default is single step increment
                one = Primitive(Token(INTEGER, 1))
                right = BinOp(var, Token(PLUS, '+'), one, env)
                post = Assign(var, Token('ASSIGN','is'), right, env)
                
        # if condition based for loop
        if begin_range is None:
            self.eat(COMMA)
            cond = self.expr_a(env)
            self.eat(COMMA)
            post = self.expr_a(env)
        
        self.eat(THEN)
        consequent = self.compound_statement(env)
        self.eat(END)
        return For(var=var,assign=assignment,range=(begin_range,end_range),cond=cond,post=post,conseq=consequent,env=env)

    def if_statement(self, env, in_else=None):
        """if_statement : IF BOOL THEN statement_list (else_statement)"""
        if in_else is None:
            in_else = False

        self.eat(IF)
        predicate = self.expr_a(env)
        self.eat(THEN)

        consequent = self.compound_statement(env)

        if self.current_token.type == ELSE:
            alternative = self.else_statement(env)
        else:
            alternative = None

        if not in_else:
            self.eat(ENDIF)

        cond = Conditional(predicate,consequent,alternative, env)

        return cond

    def else_statement(self, env):
        """else_statement : ELSE statement_list"""
        self.eat(ELSE)
        alternative = self.compound_statement(env)
        return alternative

    def boolean(self, env):
        node = self.expr_a(env)
        self.eat(BOOL)
        return node

    def compound_statement(self, env):
        """compound_statement : statement_list """
        caller = inspect.stack()[1][3]
        if caller != 'program':
            env = env.extend()

        nodes = self.statement_list(env)
        
        root = Compound(env)
        for node in nodes:
            root.children.append(node)
            
        return root
            
    def program(self, env):
        """program : compound_statement DOT"""
        self.eat(BEGIN)
        node = self.compound_statement(env)
        self.eat(END)
        self.eat(DOT)
        return node
        
    def empty(self):
        return NoOp()
        
    # NTS: order matters here
    def factor(self, env):
        token = self.current_token
        if token.type in (PLUS, MINUS):
            self.eat(token.type)
            node = UnaryOp(token, self.factor(env))
            return node
        elif token.type in (INTEGER, BOOL):
            self.eat(token.type)
            return Primitive(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr_a(env)
            self.eat(RPAREN)
            return node
        elif token.type == NOT:
            self.eat(NOT)
            node = UnaryOp(token, self.expr_a(env))
            return node
        else:
            node = self.variable(env)
            return node
    
    # highest precedence expression  
    def expr_e(self, env):
        token = self.lexer.peekToken()

        if token.type in (MULTIPLY, DIVIDE):
            return self.expr(tup=(MULTIPLY,DIVIDE), func=self.factor, env=env)
        elif token.type in (PPLUS, MMINUS):
            return self.incdec_statement(env)
        return self.factor(env)
        
    # expression with less precedence    
    def expr_d(self, env):
        """
        S -> A PLUS B
        S -> A MINUS B
        A,B -> S, INTEGER
        """
        return self.expr(tup=(PLUS,MINUS), func=self.expr_e, env=env)

    # expression with even less precedence
    def expr_c(self, env):
        """
        expr_c -> expr_d
        expr_c -> expr_c (GTHAN|LTHAN|EQUALS) expr_c
        """
        return self.expr(tup=(GTHAN,LTHAN,GTEQUALS,LTEQUALS), func=self.expr_d, env=env)

    # expression with even less precedence
    def expr_b(self, env):
        return self.expr(tup=(EQUALS,NEQUALS), func=self.expr_c, env=env)
        
    def expr_a(self, env):
        return self.expr(tup=(AND,OR),func=self.expr_b, env=env)

    # func is the next highest precedence expr function
    # and tup is a tuple containing token types to evaluate
    def expr(self, tup, func, env):
        node = func(env)

        while self.current_token.type in tup:
            token = self.current_token
            self.eat(token.type)
            node =  BinOp(left=node, op=token, right=func(env), env=env)
        return node

    def parse(self, env):
        node = self.program(env)
        if self.current_token.type != EOF:
            self.error()
            
        return node
        
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
        
    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))
        
class Interpreter(NodeVisitor):
    GLOBAL_ENV = Env()
    
    def __init__(self, parser):
        self.parser = parser
    
    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MULTIPLY:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIVIDE:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == GTHAN:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == LTHAN:
            return self.visit(node.left) < self.visit(node.right)
        elif node.op.type == GTEQUALS:
            return self.visit(node.left) >= self.visit(node.right)
        elif node.op.type == LTEQUALS:
            return self.visit(node.left) <= self.visit(node.right)
        elif node.op.type == EQUALS:
            return self.visit(node.left) == self.visit(node.right)
        elif node.op.type == AND:
            return self.visit(node.left) and self.visit(node.right)
        elif node.op.type == OR:
            return self.visit(node.left) or self.visit(node.right)
            
    def visit_Primitive(self, node):
        return node.value

    def visit_Bool(self, node):
        if node.token.type == BOOL:
            return node.value
        elif hasattr(node, 'op'):
            if node.op.type in (GTHAN,LTHAN,GTEQUALS,LTEQUALS,EQUALS,AND,OR):
                return self.visit_BinOp(node)
            elif node.op.type == NOT:
                return self.visit_UnaryOp(node)
        
    def visit_Conditional(self, node):
        if self.visit_Bool(node.pred):
            self.visit(node.conseq)
        elif node.alt is not None:
            self.visit(node.alt)

    def visit_While(self, node):
        if self.visit_Bool(node.pred):
            self.visit(node.conseq)
            self.visit(node)

    def visit_For(self, node):
        left = self.visit(node.assign)
        self.visit_For_helper(node)

    def visit_For_helper(self, node):
        if node.var is not None:
            print("here")
            print(node.var.env.vars)
            cond = self.visit(node.var) < self.visit(node.range[1])
        else:
            cond = self.visit_Bool(node.cond)
        if cond:
            self.visit(node.conseq)
            self.visit(node.post)
            self.visit_For_helper(node)

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)
        elif op == PPLUS:
            return self.visit(node.expr) + 1
        elif op == MMINUS:
            return self.visit(node.expr) - 1
        elif op == NOT:
            result = self.visit(node.expr)
            if type(result) is bool:
                return not self.visit(node.expr)
            else:
                raise Exception("'not' can only be applied to boolean values")
            
    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)
            
    def visit_NoOp(self, node):
        pass
    
    def visit_Assign(self, node):
        var_name = node.left.value
        right = self.visit(node.right)
        node.left.env.set(var_name, right)
        print(node.left.env.vars)
        return right
        
    def visit_Var(self, node):
        var_name = node.value
        val = node.env.get(var_name)
        if val is None:
            print(node.env.vars)
            raise NameError(repr(var_name))
        else:
            return val
            
    def interpret(self):
        self.visit(self.parser.parse(self.GLOBAL_ENV))
        
def main():
    import sys
    text = open(sys.argv[1], 'r').read()
    
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.interpret()
    print(interpreter.GLOBAL_ENV.vars)
        
if __name__ == '__main__':
    main()
