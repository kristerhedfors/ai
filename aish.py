import sys
import argparse
import cmd
import readline
import glob
import os.path
from pathlib import Path
import ast
import json
import time
import tempfile
import unittest
import difflib
from termcolor import colored
from pprint import pprint as pp
import openai
from pygments import highlight
from pygments.lexers import PythonLexer, GoLexer, CLexer, CppLexer, RustLexer, BashLexer
from pygments.formatters import TerminalFormatter
import logging


debug_mode = True
#debug_mode = False

logging.basicConfig(level=logging.DEBUG)
def debug(*args):
    if debug_mode:
        msg = ' '.join([str(type(arg)) for arg in args])
        logging.debug(msg)



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


class GPTAnswer(object):

    def __init__(self, question, answer, state):
        self.question = question
        self.answer = answer
        self.state = state
        self._init_highlight()
        self._highlighted_answer = None
        self._current_language = None

    def __repr__(self):
        return f"Q: {self.question}\nA: {self.answer}\nS: {self.state}"
    
    def __str__(self):
        return self.answer

    def _init_highlight(self):
        self._formatter = TerminalFormatter()
        self._inside_code_block = False
        self._language_lexer_map = {
            "python": PythonLexer(),
            "golang": GoLexer(),
            "c": CLexer(),
            "cpp": CppLexer(),
            "c++": CppLexer(),
            "bash": BashLexer(),
            "terminal": BashLexer(),
        }
    
    def _highlight(self, s, language):
        lexer = self._language_lexer_map.get(language, None)
        if lexer:
            return highlight(s, lexer, self._formatter)
        return s
    
    def get_most_likely_language(self, lines, idx):
        """ return a tuple of (language, is_code_block_delimiter)
            * the most likely language for the code block starting at idx,
              or None if idx does not start a code block.
            * boolean denoting if line is a code block delimiter or not
            Judge either by name following ``` or by most frequently occurring
            language in the previous 3 lines.
        """
        if not lines[idx].startswith('```'):
            return (self._current_language, False)
        
        if self._inside_code_block:
            self._inside_code_block = False
            self._current_language = None
            return (self._current_language, True)

        self._inside_code_block = True

        if lines[idx].startswith('```python'):
            self._current_language = 'python'
        elif lines[idx].startswith('```go'):
            self._current_language = 'golang'
        elif lines[idx].startswith('```c'):
            self._current_language = 'c'
        elif lines[idx].startswith('```cpp'): 
            self._current_language = 'cpp'
        elif lines[idx].startswith('```c++'):
            self._current_language = 'c++'
        elif lines[idx].startswith('```rust'):
            self._current_language = 'rust'
        elif lines[idx].startswith('```bash'):
            self._current_language = 'bash'
        elif lines[idx].startswith('```terminal'):
            self._current_language = 'terminal'
        else:
            # find most frequently occurring language in the previous 3 lines
            counter = {
                "python": 0,
                "golang": 0,
                " c ": 0,
                "cpp": 0,
                "c++": 0,
                "rust": 0,
                "bash": 0,
                "terminal": 0
            }
            for i in range(idx-3, idx):
                for language in counter:
                    counter[language] += len(lines[i].split(language)) - 1
            self._current_language = max(counter, key=counter.get)
        return (self._current_language, True)
    
    def get_code_blocks(self):
        ''' return a list of (language, code_block) tuples, where language is the
            language of the code block, and code_block is the joined lines of the
            code block. If language is None, the code block is plain text.
        '''
        lines = self.answer.splitlines()
        language_tagged_lines = []
        for idx, line in enumerate(lines):
            (language, is_code_block_delimiter) = self.get_most_likely_language(lines, idx)
            if not is_code_block_delimiter:
                language_tagged_lines.append((language, line))
        # now we have a list of (language, line) tuples
        # we want to split this into a list of (language, code_block) tuples
        # where code_block the joined lines of the code block

        code_blocks = []
        current_language = None # None means no code language, thus plain text
        current_block = []
        for language, line in language_tagged_lines:
            if language == current_language:
                current_block.append(line)
            else:
                stripped_block = '\n'.join(current_block).strip()
                if stripped_block:
                    code_blocks.append((current_language, stripped_block))
                current_language = language
                current_block = [line]
        stripped_block = '\n'.join(current_block).strip()
        if stripped_block:
            code_blocks.append((current_language, stripped_block))
        return code_blocks
    
    def highlight(self):
        ''' return the GPT answer, with syntax highlighting for code blocks
        '''
        if self._highlighted_answer is not None:
            return self._highlighted_answer
        lines = self.answer.splitlines()
        highlighted_lines = []
        for idx, line in enumerate(lines):
            language = self.get_most_likely_language(lines, idx)
            if language:
                highlighted_lines.append(str(language) + ' ' + self._highlight(line, language))
            else:
                highlighted_lines.append(str(language) + ' ' + line)
        self._highlighted_answer = '\n'.join(highlighted_lines)
        return self._highlighted_answer


# highlight testing
if 0:
    answer1 = '''
Hey look at this dope python code I wrote:
```python
def test1():
    print("Hello1")
```
nice huh? here we to some terminal stuff and in the terminal we invoke a python script
since we love
look here
```
$ find . -name '*.py' | xargs grep 'def' | ./aish.py update_function aish.py test1
```
    '''
    def test_highlight():
        answer = GPTAnswer("", answer1, {})
        # assert that the first code block is python
        print(answer.highlight())

    def test_get_code_blocks():
        answer = GPTAnswer("", answer1, {})
        pp(answer.get_code_blocks())

    #test_highlight()
    test_get_code_blocks()
    sys.exit()


class Test_GPTAnswer(unittest.TestCase):
    answer1 = '''
Hey look at this dope python code I wrote:
```python
def test1():
    print("Hello1")
```
nice huh?
```'''

    answer2 = '''
Hey look at this dope python code I wrote:
```python
def test1():
    print("Hello1")
```
nice huh? here we to some terminal stuff and in the terminal we invoke a python script
since we love
look here
```
$ find . -name '*.py' | xargs grep 'def' | ./aish.py update_function aish.py test1
```
    '''
    def test_get_code_blocks(self):
        answer = GPTAnswer("", self.answer1, {})
        code_blocks = answer.get_code_blocks()
        self.assertEqual(len(code_blocks), 3)
        self.assertEqual(code_blocks[0][0], None)
        self.assertEqual(code_blocks[1][0], "python")
        self.assertEqual(code_blocks[2][0], None)
        self.assertEqual(code_blocks[0][1], 'Hey look at this dope python code I wrote:')
        self.assertEqual(code_blocks[1][1], 'def test1():\n    print("Hello1")')
        self.assertEqual(code_blocks[2][1], 'nice huh?')

    def test_get_code_blocks2(self):
        answer = GPTAnswer("", self.answer2, {})
        code_blocks = answer.get_code_blocks()
        self.assertEqual(len(code_blocks), 4)
        self.assertEqual(code_blocks[0][0], None)
        self.assertEqual(code_blocks[1][0], 'python')
        self.assertEqual(code_blocks[2][0], None)
        self.assertEqual(code_blocks[3][0], 'terminal')
        self.assertEqual(code_blocks[0][1], 'Hey look at this dope python code I wrote:')
        self.assertEqual(code_blocks[1][1], 'def test1():\n    print("Hello1")')
        self.assertEqual(code_blocks[2][1], 'nice huh? here we to some terminal stuff and in the terminal we invoke a python script\nsince we love\nlook here')
        self.assertEqual(code_blocks[3][1], '$ find . -name \'*.py\' | xargs grep \'def\' | ./aish.py update_function aish.py test1')

class GPT(object):

    state_dir = Path.home() / ".ai"
    curr_state = None

    def __init__(self):
        self.state_dir.mkdir(exist_ok=True)
        self.curr_state = "state-{}".format(int(time.time()))
        self.curr_state_file = self.state_dir / self.curr_state
        if not self.curr_state_file.exists():
            with open(self.curr_state_file, 'w') as f:
                json.dump([], f)

    def use_state(self, state_name):
        self.curr_state = state_name

    system_role_desc = 'You are a helpful assistant.'

    def set_system_role(self, desc):
        self.system_role_desc = desc

    def get_system_role(self):
        return self.system_role_desc

    def list_states(self):
        return [str(p.name) for p in self.state_dir.glob('*') if p.is_file()]

    def load_state(self, state_name=None):
        if state_name is None:
            state_name = self.curr_state
        state_file = self.state_dir / state_name
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        else:
            return []

    def save_state(self, save_name):
        curr_file = self.state_dir / self.curr_state
        save_file = self.state_dir / self.save_name
        if save_file.exists():
            raise Exception("State already exists")
        print(f"Saving state to {save_file} from {curr_file}")
        with open(save_file, 'w') as fw:
            with open(curr_file, 'r') as fr:
                fw.write(fr.read())

    def update_state(self, state):
        state_file = self.state_dir / self.curr_state
        with open(state_file, 'w') as f:
            json.dump(state, f)

    def clear_state(self):
        state_file = self.state_dir / self.curr_state
        if state_file.exists():
            os.remove(state_file)

    def load_model(self):
        model_file = Path.home() / ".aimodel"
        if model_file.exists():
            with open(model_file, 'r') as f:
                return f.read().strip()
        else:
            return "gpt-3.5-turbo"

    def ask(self, question):
        global debug_mode
        state = self.load_state()
        model = self.load_model()

        messages = [{"role": "system", "content": self.system_role_desc}]
        for h in state:
            messages.append(h)
        messages.append({"role": "user", "content": question})

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages
                )
                break
            except openai.error.RateLimitError:
                print("Rate limit exceeded, retrying...")
                time.sleep(5)

        if debug_mode:
            print("Query: ", json.dumps(messages, indent=4))
            print("Response: ", json.dumps(response, indent=4))

        answer = response['choices'][0].message.content
        state.append({"role": "user", "content": question})
        state.append({"role": "assistant", "content": answer})
        self.update_state(state)
        return GPTAnswer(question, answer, state)


class InteractiveShell(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = '> '

        self.parser = argparse.ArgumentParser()
        subparsers = self.parser.add_subparsers()

        # ask
        parser_ask = subparsers.add_parser("ask")
        parser_ask.add_argument("question", nargs=argparse.REMAINDER)
        parser_ask.set_defaults(func=self.do_ask)


        # summarize_file
        parser_summarize_file = subparsers.add_parser("summarize_file")
        parser_summarize_file.add_argument("filename", type=str)
        parser_summarize_file.set_defaults(func=self.do_summarize_file)

        # update-file
        parser_update_file = subparsers.add_parser("update_file")
        parser_update_file.add_argument("filename", type=str)
        parser_update_file.add_argument("instruction", nargs=argparse.REMAINDER)
        parser_update_file.set_defaults(func=self.do_update_file)
    
        # update-function
        parser_update_function = subparsers.add_parser("update_function")
        parser_update_function.add_argument("filename", type=str)
        parser_update_function.add_argument("function_name", type=str)
        parser_update_function.set_defaults(func=self.do_update_function)

        # run
        parser_run = subparsers.add_parser("run")
        parser_run.add_argument("executable", type=str)
        parser_run.add_argument("arguments", nargs=argparse.REMAINDER)
        parser_run.set_defaults(func=self.do_run)

        self.completion_map = {
            "ask": ["What", "When", "Where", "Why", "How"],
            "run": glob.glob('*'),
            "summarize_file": glob.glob('*'),
            "update_file": glob.glob('*'),
            "update_function": glob.glob('*')
        }

    #
    # ask
    #
    def do_ask(self, args):
        '''Asks a question to the AI.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'ask {args}'.split())
        question = ' '.join(args.question)
        answer = GPT().ask(question)
        print(answer)
    
    def complete_ask(self, text, line, begidx, endidx):
        '''Suggest three alternative words for the next word of the command, fetched from GPT().ask(command)'''
        if line.strip() == 'ask' or len(line.split()) == 2:
            comp_list = [i for i in self.completion_map["ask"] if i.startswith(text)] 
            if comp_list:
                return comp_list
        return []
        #question = ' '.join(args.question)
        pp('-------<asking>-------')
        #answer = GPT().ask('Suggest six completion alternatives for the following sentence: ' + question)
        # see complete_update_function to figure out completion, based on args
        pp('-------<answer>----')
        pp(answer)
        pp('-------<text>------')
        pp(text)
        pp('-------000000------')
        pp(answer)
        words = answer.split()
        return ['asd', 'qwe']
        if text:
            return [i for i in words if i.startswith(text)]
        else:
            return words

    
    #
    # summarize_file
    #
    def do_summarize_file(self, args):
        '''Summarizes the contents of a file.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'summarize_file {args}'.split())
        with open(args.filename) as f:
            contents = f.read()
        question = f"Summarize the file `{args.filename}`, file contents pasted below:\n\n{contents}"
        answer = GPT().ask(question)
        print(answer)
    
    def complete_summarize_file(self, text, line, begidx, endidx):
        if text:
            return [
                f for f in glob.glob(text+'*') 
                if os.path.isfile(f)
            ]
        else:
            return glob.glob('*')

    #
    # update_file
    #
    def do_update_file(self, args):
        '''Updates the modification time of a file to the current time.'''
        if isinstance(args, str):
            args = self.parser.parse_args(f'update_file {args}'.split())
        with open(args.filename) as f:
            contents = f.read()
        question = f"The file `{args.filename}`, file contents pasted below double line breaks at end of message,"
        question = f" according to the following instructions: `{args.instruction}`"
        question += f"\n\n{contents}" 
        answer = GPT().ask(question)
        print(answer)

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

    #
    # update-function
    #
    def do_update_function(self, arg):
        '''Prints the source of a function in a Python file.'''
        if isinstance(arg, str):
            args = self.parser.parse_args(f'update_function {arg}'.split())
        else:
            args = arg
        with open(args.filename) as f:
            module = ast.parse(f.read())
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            if function.name == args.function_name:
                print(f"Found function {args.function_name} in file {args.filename}")

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
    #
    # run
    #
    def do_run(self, arg):
        '''Runs a Python file.'''
        if isinstance(arg, str):
            args = self.parser.parse_args(f'run {arg}'.split())
        else:
            args = arg
        # execute using subprocess
        p = subprocess.run([args.executable] + args.arguments, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        print(p.stdout.decode('utf-8'))
    
    def complete_run(self, text, line, begidx, endidx):
        args = line.split()
        if text:
            if len(args) >= 2:
                return [
                    f for f in glob.glob(text+'*') 
                    if os.path.isfile(f)
                ]
            else:
                return [
                    f for f in glob.glob(text+'*') 
                ]
        else:
            return glob.glob('*')


import subprocess
import pexpect

if 0: 
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
