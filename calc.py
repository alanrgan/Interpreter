INTEGER, BOOL = 'INTEGER', 'BOOL'
PLUS, MINUS, MULTIPLY, EOF = 'PLUS', 'MINUS', 'MULTIPLY', 'EOF'
DIVIDE = 'DIVIDE'
LPAREN, RPAREN = 'LPAREN', 'RPAREN'
BEGIN, END, DOT = 'BEGIN', 'END', 'DOT'
ID, ASSIGN, SEMI = 'ID', 'ASSIGN', 'SEMI'
PPLUS, MMINUS = 'PPLUS', 'MMINUS'
PEQUALS, MEQUALS = 'PEQUALS', 'MEQUALS'
IF, ELSE, THEN, ENDIF = 'IF', 'ELSE', 'THEN', 'ENDIF'
GTHAN,LTHAN,EQUALS = 'GTHAN', 'LTHAN', 'EQUALS'

import inspect
import unicodedata

class AST(object):
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Primitive(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
        
class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr
        
class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self):
        self.children = []
        
class Conditional(AST):
    """Represents a conditional fork"""
    def __init__(self, pred, conseq, alt=None):
        self.pred = pred
        self.conseq = conseq
        self.alt = alt

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class NoOp(AST):
    pass

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
    'BEGIN': Token('BEGIN', 'BEGIN'),
    'END': Token('END', 'END'),
    'is': Token('ASSIGN', 'IS'),  
    'if': Token('IF', 'if'),
    'else': Token('ELSE', 'else'),
    'endif': Token('ENDIF', 'endif'),
    'then': Token('THEN', 'then'),
    'true': Token('BOOL', True),
    'false': Token('BOOL', False),
    'equals': Token('EQUALS', 'equals'),
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
                    self.col_num = 1
                self.skip_whitespace()
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
                
            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '>':
                self.advance()
                return Token(GTHAN, '>')

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
        token_val = self.current_token.value
        exp_msg = '' if expect is None else ', Expected: ' + expect
        raise Exception('Invalid syntax on line '+lnum+', col '+cnum+': '+token_val+exp_msg)
        
    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.prev_token = self.current_token
            self.current_token = self.lexer.get_next_token()
        else:
            self.error(token_type)

    def variable(self):
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr_a()
        node = Assign(left, token, right)
        return node
        
    def incdec_statement(self):
        """
        incdec is same as varASSIGN(varADD1)
        """
        left = self.variable()
        token = self.current_token
        one = Primitive(Token(INTEGER, 1))
        if token.type == PPLUS:
            self.eat(PPLUS)
            right = BinOp(left, Token(PLUS, '+'), one)
        else:
            self.eat(MMINUS)
            right = BinOp(left, Token(MINUS, '-'), one)
        node = Assign(left, token, right)
        return node
        
    def incdec_assign_statement(self):
        left = self.variable()
        token = self.current_token
        if token.type == PEQUALS:
            self.eat(PEQUALS)
            right = BinOp(left, Token(PLUS, '+'), self.expr_b())
        elif token.type == MEQUALS:
            self.eat(MEQUALS)
            right = BinOp(left, Token(MINUS, '-'), self.expr_b())
        node = Assign(left, token, right)
        return node

    def statement(self):
        if self.current_token.type == BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == IF:
            in_else = self.prev_token.type == ELSE
            node = self.if_statement(in_else=in_else)
        elif self.current_token.type == ID:
            next_token = self.lexer.peekToken()
            if next_token.type == ASSIGN:
                node = self.assignment_statement()
            elif next_token.type in (PPLUS, MMINUS):
                node = self.incdec_statement()
            elif next_token.type in (PEQUALS, MEQUALS):
                node = self.incdec_assign_statement()
            else:
                node = self.empty()
        else:
            node = self.empty()
        return node

    def statement_list(self):
        node = self.statement()
        results = [node]
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results.append(self.statement())
            
        if self.current_token.type == ID:
            self.error()
            
        return results

    def if_statement(self, in_else=None):
        """if_statement : IF BOOL THEN statement_list (else_statement)"""
        if in_else is None:
            in_else = False

        self.eat(IF)
        predicate = self.expr_a()
        self.eat(THEN)
        nodes = self.statement_list()

        consequent = Compound()
        for node in nodes:
            consequent.children.append(node)

        if self.current_token.type == ELSE:
            alternative = self.else_statement()
        else:
            alternative = None

        if not in_else:
            self.eat(ENDIF)

        cond = Conditional(predicate,consequent,alternative)

        return cond

    def else_statement(self):
        """else_statement : ELSE statement_list"""
        self.eat(ELSE)
        alt_nodes = self.statement_list()
        alternative = Compound()

        for i,node in enumerate(alt_nodes):
            alternative.children.append(node)
        return alternative

    def boolean(self):
        node = self.expr_a()
        self.eat(BOOL)
        return node

    def compound_statement(self):
        """compound_statement : BEGIN statement_list END"""
        self.eat(BEGIN)
        nodes = self.statement_list()
        self.eat(END)
        root = Compound()
        for node in nodes:
            root.children.append(node)
            
        return root
            
    def program(self):
        """program : compound_statement DOT"""
        node = self.compound_statement()
        self.eat(DOT)
        return node
        
    def empty(self):
        return NoOp()
        
    # NTS: order matters here
    def factor(self):
        token = self.current_token
        if token.type in (PLUS, MINUS):
            self.eat(token.type)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == INTEGER:
            self.eat(INTEGER)
            return Primitive(token)
        elif token.type == BOOL:
            self.eat(BOOL)
            return Primitive(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr_a()
            self.eat(RPAREN)
            return node
        else:
            node = self.variable()
            return node

    def bool_factor(self):
        pass
    
    # highest precedence expression  
    def expr_c(self):
        node = self.factor()
        
        while self.current_token.type in (MULTIPLY, DIVIDE, PPLUS, MMINUS):
            token = self.current_token
            self.eat(token.type)
            if token.type in (MULTIPLY, DIVIDE):
                node = BinOp(left=node, op=token, right=self.factor())
            elif token.type in(PPLUS, MMINUS):
                node = UnaryOp(op=token, expr=node)
            
        return node
        
    # expression with second least precedence    
    def expr_b(self):
        """
        S -> A PLUS B
        S -> A MINUS B
        A,B -> S, INTEGER
        """
        node = self.expr_c()
        while self.current_token.type in (PLUS,MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            node = BinOp(left=node, op=token, right=self.expr_c())
        return node

    # expression with least precedence
    def expr_a(self):
        """
        bool_expr -> bool_expr EQUALS bool_expr
        bool_expr -> expr (GTHAN|LTHAN|EQUALS) expr
        """
        node = self.expr_b()
        while self.current_token.type in (GTHAN, LTHAN, EQUALS):
            token = self.current_token
            if token.type == GTHAN:
                self.eat(GTHAN)
            elif token.type == LTHAN:
                self.eat(LTHAN)
            elif token.type == EQUALS:
                self.eat(EQUALS)

            node = BinOp(left=node, op=token, right=self.expr_b())
        return node
        
    def parse(self):
        node = self.program()
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
    GLOBAL_SCOPE = {}    
    
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
        elif node.op.type == EQUALS:
            return self.visit(node.left) == self.visit(node.right)
            
    def visit_Primitive(self, node):
        return node.value

    def visit_Bool(self, node):
        if node.token.type == BOOL:
            return node.value
        elif hasattr(node, 'op') and node.op.type in (GTHAN,LTHAN,EQUALS):
            return self.visit_BinOp(node)
        
    def visit_Conditional(self, node):
        if self.visit_Bool(node.pred):
            self.visit_Compound(node.conseq)
        elif node.alt is not None:
            self.visit_Compound(node.alt)

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
            
    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)
            
    def visit_NoOp(self, node):
        pass
    
    def visit_Assign(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)
        
    def visit_Var(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val
            
    def interpret(self):
        self.visit(self.parser.parse())
        
def main():
    import sys
    text = open(sys.argv[1], 'r').read()
    
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.interpret()
    print(interpreter.GLOBAL_SCOPE)
        
if __name__ == '__main__':
    main()
