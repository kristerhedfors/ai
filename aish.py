import argparse
import cmd
import readline
import glob
import os.path
import ast
import tempfile
import unittest
from pprint import pprint as pp


import difflib
from termcolor import colored

def display_diff(orig_string, updated_string):
    """
    Displays the syntax highlighted diff-style changes from orig_string to updated_string.
    """
    diff = difflib.unified_diff(orig_string.splitlines(), updated_string.splitlines())

    # Get the unified diff text in a string format
    diff_text = '\n'.join(list(diff)[2:])

    # Break the diff text into separate lines
    diff_text = diff_text.split('\n')

    # Parse the lines and highlight the differences
    highlighted_lines = []
    for line in diff_text:
        # Check if the line describes a change
        if line.startswith('+'):
            highlighted_lines.append(colored(line, 'green'))
        elif line.startswith('-'):
            highlighted_lines.append(colored(line, 'red'))
        else:
            highlighted_lines.append(line)

    # Join the highlighted lines back into a single string
    highlighted_text = '\n'.join(highlighted_lines)

    # Print the highlighted text
    print(highlighted_text)


class PyMod(object):

    def __init__(self, filename):
        self.filename = filename

    def extract_function(self, func_name):
        '''Extracts the source code of a function from a python file.
        Known bug: functions ending at the end of the file missing \\n\\n
        will not be found
        '''
        # extract source code of function from filename
        # find the function definition
        with open(self.filename, 'r') as f:
            content = f.read()
        assert len(content) > 0
        func_def = 'def {}'.format(func_name)
        func_start = content.find(func_def)
        if func_start == -1:
            raise Exception(content)
            raise Exception(f"Could not find function definition for {func_name}")
        # find the end of the function
        func_end = content.find('\n\n', func_start)
        if func_end == -1:
            raise Exception(f"Could not find end of function definition for {func_name}")
        # extract the function
        func = content[func_start:func_end]
        return func, func_start, func_end

    def replace_function(self, func_name, new_func):
        func, func_start, func_end = self.extract_function(func_name)
        self._replace_function(func_name, func, func_start, func_end, new_func)
    
    def _replace_function(self, func_name, func, func_start, func_end, new_func):
        with open(self.filename, 'r') as f:
            content = f.read()
        if not content.index(func) == func_start:
            raise Exception("Function definition does not match expected location")
        # replace func with new_func
        content = content[:func_start] + new_func + content[func_end:]
        with open(self.filename, 'w') as f:
            f.write(content)
            f.flush()
    
    def extract_class(self, class_name):
        '''Extracts the source code of a class from a python file.
        Known bug: classes ending at the end of the file missing \\n\\n
        will not be found
        '''
        # extract source code of class from filename
        # find the class definition
        with open(self.filename, 'r') as f:
            content = f.read()
        assert len(content) > 0
        class_def = 'class {}'.format(class_name)
        class_start = content.find(class_def)
        if class_start == -1:
            raise Exception(f"Could not find class definition for {class_name}")
        # find the end of the class
        class_end = content.find('\n\n', class_start)
        if class_end == -1:
            raise Exception(f"Could not find end of class definition for {class_name}")
        # extract the class
        class_ = content[class_start:class_end]
        return class_, class_start, class_end
    
    def replace_class(self, class_name, new_class):
        class_, class_start, class_end = self.extract_class(class_name)
        self._replace_class(class_name, class_, class_start, class_end, new_class)
    
    def _replace_class(self, class_name, class_, class_start, class_end, new_class):
        with open(self.filename, 'r') as f:
            content = f.read()
        if not content.index(class_) == class_start:
            raise Exception("Class definition does not match expected location")
        # replace class with new_class
        content = content[:class_start] + new_class + content[class_end:]
        with open(self.filename, 'w') as f:
            f.write(content)
            f.flush()


class Test_PyMod(unittest.TestCase):

    three_functions = b'''
def test1():
    print("Hello1")

def test2():
    print("Hello2")

def test3():
    print("Hello3")

'''
    two_classes = b'''
class Test1:
    def __init__(self):
        print("Hello1")

class Test2:
    def __init__(self):
        print("Hello2")
    
    def add(self, x, y):
        return x + y

'''
    def setUp(self):
        # create a temporary python file containing three functions
        # three_functions 
        self.three_functions_file = tempfile.NamedTemporaryFile(suffix='.py')
        self.three_functions_file.write(self.three_functions)
        self.three_functions_file.flush()
        # two_classes
        self.two_classes_file = tempfile.NamedTemporaryFile(suffix='.py')
        self.two_classes_file.write(self.two_classes)
        self.two_classes_file.flush()
    
    def tearDown(self):
        # delete the temporary python file
        self.three_functions_file.close()

    def test_extract_function(self):
        mod = PyMod(self.three_functions_file.name)
        # 1
        func, func_start, func_end = mod.extract_function('test1')
        self.assertEqual(func, 'def test1():\n    print("Hello1")')
        # 2
        func, func_start, func_end = mod.extract_function('test2')
        self.assertEqual(func, 'def test2():\n    print("Hello2")')
        # 3
        func, func_start, func_end = mod.extract_function('test3')
        self.assertEqual(func, 'def test3():\n    print("Hello3")')
    
    def test_replace_function(self):
        mod = PyMod(self.three_functions_file.name)
        mod.replace_function('test2', 'def test2():\n    print("Goodbye2")')
        # 1
        mod.replace_function('test1', 'def test1():\n    print("Goodbye1")')
        func, func_start, func_end = mod.extract_function('test1')
        self.assertEqual(func, 'def test1():\n    print("Goodbye1")')
        # 2
        mod.replace_function('test2', 'def test2():\n    print("Goodbye2")')
        func, func_start, func_end = mod.extract_function('test2')
        self.assertEqual(func, 'def test2():\n    print("Goodbye2")')
        # 3
        mod.replace_function('test3', 'def test3():\n    print("Goodbye3")')
        func, func_start, func_end = mod.extract_function('test3')
        self.assertEqual(func, 'def test3():\n    print("Goodbye3")')

    def test_extract_class(self):
        mod = PyMod(self.two_classes_file.name)
        # 1
        class_, class_start, class_end = mod.extract_class('Test1')
        self.assertEqual(class_, 'class Test1:\n    def __init__(self):\n        print("Hello1")')
        # 2
        class_, class_start, class_end = mod.extract_class('Test2')
        self.assertEqual(class_, 'class Test2:\n    def __init__(self):\n        print("Hello2")\n    \n    def add(self, x, y):\n        return x + y')


class InteractiveShell(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = '> '

        self.parser = argparse.ArgumentParser()
        subparsers = self.parser.add_subparsers()

        parser_greet = subparsers.add_parser("greet")
        parser_greet.add_argument("name", type=str)
        parser_greet.set_defaults(func=self.do_greet)

        parser_sum = subparsers.add_parser("sum")
        parser_sum.add_argument("x", type=int)
        parser_sum.add_argument("y", type=int)
        parser_sum.set_defaults(func=self.do_sum)

        parser_update_file = subparsers.add_parser("update-file")
        parser_update_file.add_argument("filename", type=str)
        parser_update_file.set_defaults(func=self.do_update_file)
    
        parser_update_function = subparsers.add_parser("update-function")
        parser_update_function.add_argument("filename", type=str)
        parser_update_function.add_argument("function_name", type=str)
        parser_update_function.set_defaults(func=self.do_update_function)

        self.completion_map = {
            "greet": ["Alice", "Bob", "Charlie"],
            "sum": [str(i) for i in range(10)],
            "update-file": glob.glob('*'),
            "update-function": glob.glob('*')
        }

    def do_greet(self, args):
        '''Outputs a greeting to the given name.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'greet {args}'.split())
        print(f"Hello, {args.name}!")

    def complete_greet(self, text, line, begidx, endidx):
        if text:
            return [i for i in self.completion_map["greet"] if i.startswith(text)]
        else:
            return self.completion_map["greet"]

    def do_sum(self, args):
        '''Outputs the sum of the two arguments.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'sum {args}'.split())
        print(args.x + args.y)

    def complete_sum(self, text, line, begidx, endidx):
        if text:
            return [i for i in self.completion_map["sum"] if i.startswith(text)]
        else:
            return self.completion_map["sum"]

    def do_update_file(self, args):
        '''Updates the modification time of a file to the current time.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'update-file {args}'.split())
        os.utime(args.filename, None)
        print(f"Updated file: {args.filename}")

    def complete_update_file(self, text, line, begidx, endidx):
        if text:
            return [
                f for f in glob.glob(text+'*') 
                #if os.path.isfile(f)
            ]
        else:
            return glob.glob('*')

    def parseline(self, line):
        self.last_command = line
        return super().parseline(line)

    # Your existing do_ methods...

    def do_update_function(self, arg):
        '''Prints the source of a function in a Python file.'''
        if isinstance(arg, str):
            args = self.parser.parse_args(f'update-function {arg}'.split())
        else:
            args = arg
        with open(args.filename) as f:
            module = ast.parse(f.read())
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            if function.name == args.function_name:
                print(f"Found function {args.function_name} in file {args.filename}")

    # Your existing complete_ methods...

    def complete_update_function(self, text, line, begidx, endidx):
        args = line.split()
        if len(args) > 2 and os.path.isfile(args[1]):
            filename = args[1]
            with open(filename) as f:
                module = ast.parse(f.read())
            functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
            return [function for function in functions if function.startswith(text)]
        else:
            return [
                f for f in glob.glob(text+'*')
                if os.path.isfile(f)
            ]


import subprocess
import pexpect

class Test_InteractiveShell(unittest.TestCase):
    def test_sum_command_line(self):
        output = subprocess.check_output(["python3", "aish.py", "sum", "1", "2"])
        self.assertEqual(output.strip(), b'3')

    def test_greet_command_line(self):
        output = subprocess.check_output(["python3", "aish.py", "greet", "Alice"])
        self.assertEqual(output.strip(), b'Hello, Alice!')

    def test_sum_interactive_shell(self):
        child = pexpect.spawn('python3 aish.py')
        child.sendline('sum 1 2')
        child.expect('3')
        child.sendline('quit')

    def test_greet_interactive_shell(self):
        child = pexpect.spawn('python3 aish.py')
        child.sendline('greet Alice')
        child.expect('Hello, Alice!')
        child.sendline('quit')


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        unittest.main()
        sys.exit()

    if len(sys.argv) > 1:
        # Command line mode
        shell = InteractiveShell()
        shell.parser.parse_args(sys.argv[1:]).func(shell.parser.parse_args(sys.argv[1:]))
    else:
        # Interactive mode
        InteractiveShell().cmdloop()

