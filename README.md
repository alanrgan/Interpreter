# Interpreter
This is an interpreter for a simple programming language I created.

To run a program written in this language, run:
```
python calc.py program.txt
```

Here is a syntax and usage overview:

### Starting a program
All programs must be encapsulated in a `BEGIN` and `END.` statement:
```
BEGIN
    print("Hello World!")
END.
```

Every statement must end with a semicolon unless it is directly before an 'END' statement

### Comments
Single line comments use a single dollar sign, '$', while block comments are encapsulated by two dollar signs, '$$'.
Block comments cannot exist within a statement

Example:
```
BEGIN
    $ this is a single line comment
    $$
    this is a block comment
    that spans multiple lines
    $$
    $ the following statement is invalid
    1 + $$2$$ 3 
END.
```
### Arithmetic and Boolean Operators
```
BEGIN
    1+1; $ 2
    1-1; $ 0
    1/1; $ 1
    true and true; $ true
    true and false; $ false
    true or false; $ true
    2+3 equals 1+4; $ true
END.
```

### Variables and Assignment Operators
Assignment is denoted by the keyword `is`
```
BEGIN
    a is 2;
    b is a; $ b is 2
    c is d is 5; $ multiple variables can be assigned the same value in one statement
    
    $ arithmetic assignment operators
    a++;
    a--;
    a+=2;
    a-=2;
END.
```

### Conditionals
The syntax for if statements is: `if bool_expr then` followed by a series of statements, then `end;`
An else statement may be included as well.
```
BEGIN
    if 2+3 equals 1+4 then
        $ do stuff here
    else
        $ do stuff here instead
    end;
END.
```

### For loops
There are two kinds of for loops in this language.

The for loops have the following syntax:
```
$ first for loop
for var_name is begin..end [, post_expression] then
    statement(s);
end;

$ second for loop
for var_name is begin, cond, step then
    statement(s);
end;
```
The first for loop assigns the value `begin` to `var_name` and evaluates `post_expression` after the statements are evaluated.
If `post_expression` is not specified, the default statement is set as `var_name++`.

The second for loop initializes `var_name` to `begin` and checks if `cond` is true at every iteration, evaluating `step` at the
end of each iteration.

### Functions
Functions may be defined and called anywhere within the current scope
```
BEGIN
    say_hello_world();
    
    $ function definition
    say_hello_world::()
        print("Hello World!");
    end;
    
    say_hello_world()
END.
```
