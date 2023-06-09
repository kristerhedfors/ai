#!/usr/bin/env python3
import sys
import argparse
import cmd
import readline
import glob
from datetime import datetime
import os.path
from pathlib import Path
import subprocess
import ast
import json
import time
import tempfile
import unittest
import difflib
from termcolor import colored
from pprint import pprint as pp
import openai
import openai.error
from pygments import highlight
from pygments.lexers import PythonLexer, GoLexer, CLexer, CppLexer, RustLexer, BashLexer
from pygments.formatters import TerminalFormatter
import logging
import traceback
import pdb


logging.basicConfig(level=logging.INFO)

def debug(*args):
    msg = ' '.join([str(arg) for arg in args])
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
    ''' 
    Represents a python module. Provides methods to read and write source code of functions and classes in the module.
    NOT THREAD SAFE

    All write methods will back up the original file to f"filename.bak.%H%M%S" before writing.
    '''

    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        with open(self.file_path, 'r') as file:
            return file.read()
    
    def readlines(self):
        with open(self.file_path, 'r') as file:
            return file.readlines()
    
    def write(self, source):
        ''' writes source to file_path, backing up the original file to f"filename.bak.%H%M%S" '''
        backup_file_path = self.file_path + f".bak.{time.strftime('%H%M%S')}"
        with open(backup_file_path, 'w') as file:
            file.write(self.read())
        with open(self.file_path, 'w') as file:
            file.write(source)

    def get_source_of_node(self, node_name, node_type):
        source = self.read()
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, node_type) and node.name == node_name:
                node_source = source.split('\n')[node.lineno - 1 : node.end_lineno]
                return '\n'.join(node_source) + '\n'
        raise ValueError(f"No {node_type} '{node_name}' found in {self.file_path}")
    
    def get_source_of_function(self, func_name):
        ''' returns source code of func_name in file_path'''
        return self.get_source_of_node(func_name, ast.FunctionDef)
    
    def replace_node(self, node_type, node_name, new_node_source):
        source_lines = self.readlines()
        module = ast.parse(''.join(source_lines))
        for node in module.body:
            if isinstance(node, node_type) and node.name == node_name:
                start_lineno = node.lineno
                end_lineno = node.end_lineno
                break
        else:
            raise ValueError(f"No {node_type} '{node_name}' found in {self.file_path}")
        new_source_lines = source_lines[:start_lineno - 1] + [new_node_source] + source_lines[end_lineno:]
        self.write(''.join(new_source_lines))

    def replace_function(self, func_name, new_func_source):
        return self.replace_node(ast.FunctionDef, func_name, new_func_source)
    
    def get_source_of_class(self, class_name):
        ''' returns source code of class_name in file_path'''
        return self.get_source_of_node(class_name, ast.ClassDef)

    def replace_class(self, class_name, new_class_source):
        return self.replace_node(ast.ClassDef, class_name, new_class_source)
    
    def replace_code(self, new_code):
        '''for each node found through ast in new_code, replace the corresponding node in file_path'''
        source_lines = self.readlines()
        new_module = ast.parse(new_code)
        old_module = ast.parse(''.join(source_lines))
        for new_node in new_module.body:
            if isinstance(new_node, ast.FunctionDef):
                for old_node in old_module.body:
                    if isinstance(old_node, ast.FunctionDef) and old_node.name == new_node.name:
                        start_lineno = old_node.lineno
                        end_lineno = old_node.end_lineno
                        new_source_lines = source_lines[:start_lineno - 1] + [ast.unparse(new_node)] + source_lines[end_lineno:]
                        source_lines = new_source_lines
                        break
                else:
                    raise ValueError(f"No function '{new_node.name}' found in {self.file_path}")
            elif isinstance(new_node, ast.ClassDef):
                for old_node in old_module.body:
                    if isinstance(old_node, ast.ClassDef) and old_node.name == new_node.name:
                        start_lineno = old_node.lineno
                        end_lineno = old_node.end_lineno
                        new_source_lines = source_lines[:start_lineno - 1] + [ast.unparse(new_node)] + source_lines[end_lineno:]
                        source_lines = new_source_lines
                        break
                else:
                    raise ValueError(f"No class '{new_node.name}' found in {self.file_path}")
        self.write(''.join(source_lines))


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
    odd_class = b'''
asd=123
class OddClass:
 def yay(self): pass

 def yo(self):
  pass


 def wayno(self):
  pass

foo='bar'
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
        # odd_class
        self.odd_class_file = tempfile.NamedTemporaryFile(suffix='.py')
        self.odd_class_file.write(self.odd_class)
        self.odd_class_file.flush()

    
    def tearDown(self):
        # delete the temporary python file
        self.three_functions_file.close()

    def test_get_source_of_function(self):
        mod = PyMod(self.three_functions_file.name)
        # 1
        func_source = mod.get_source_of_function('test1')
        self.assertEqual(func_source, 'def test1():\n    print("Hello1")\n')
        # 2
        func_source = mod.get_source_of_function('test2')
        self.assertEqual(func_source, 'def test2():\n    print("Hello2")\n')
        # 3
        func_source = mod.get_source_of_function('test3')
        self.assertEqual(func_source, 'def test3():\n    print("Hello3")\n')
    
    def test_replace_blankspace_side_effects(self):
        mod = PyMod(self.three_functions_file.name)
        # 1
        mod.replace_function('test1', mod.get_source_of_function('test1'))
        # TODO complete


    def test_replace_function(self):
        mod = PyMod(self.three_functions_file.name)
        # 1
        mod.replace_function('test1', 'def test1():\n    print("Goodbye1")')
        func_source = mod.get_source_of_function('test1')
        self.assertEqual(func_source, 'def test1():\n    print("Goodbye1")\n')
        # 2
        mod.replace_function('test2', 'def test2():\n    print("Goodbye2")')
        func_source = mod.get_source_of_function('test2')
        self.assertEqual(func_source, 'def test2():\n    print("Goodbye2")\n')
        # 3
        mod.replace_function('test3', 'def test3():\n    print("Goodbye3")')
        func_source = mod.get_source_of_function('test3')
        self.assertEqual(func_source, 'def test3():\n    print("Goodbye3")\n')

    def test_get_source_of_class(self):
        mod = PyMod(self.two_classes_file.name)
        # 1
        class_source = mod.get_source_of_class('Test1')
        self.assertEqual(class_source, 'class Test1:\n    def __init__(self):\n        print("Hello1")\n')
        class_source = mod.get_source_of_class('Test2')
        self.assertEqual(class_source, 'class Test2:\n    def __init__(self):\n        print("Hello2")\n    \n    def add(self, x, y):\n        return x + y\n')
    
    def test_replace_class(self):
        mod = PyMod(self.two_classes_file.name)
        # 1
        mod.replace_class('Test1', 'class Test1:\n    def __init__(self):\n        print("Goodbye1")')
        class_source = mod.get_source_of_class('Test1')
        self.assertEqual(class_source, 'class Test1:\n    def __init__(self):\n        print("Goodbye1")\n')
    
    def _test_replace_code(self):
        mod = PyMod(self.two_classes_file.name)
        # 1
        new_code = self.two_classes.replace(b'Hello', b'Goodbye')
        mod.replace_code(new_code)
        mod = PyMod(self.two_classes_file.name)  # reload after file update
        with open(self.two_classes_file.name, 'r') as file:
            debug('\n' + file.read())
        class_source = mod.get_source_of_class('Test1')
        self.assertEqual(class_source, 'class Test1:\n    def __init__(self):\n        print("Goodbye1")\n')
        class_source = mod.get_source_of_class('Test2')
        self.assertEqual(class_source, 'class Test2:\n    def __init__(self):\n        print("Goodbye2")\n')


class LLMAnswer(object):

    def __init__(self, question, answer, state, state_name, default_language='python'):
        self.question = question
        self.answer = answer
        self.state = state
        self.state_name = state_name
        self.default_language = default_language
        self._init_highlight()
        self._highlighted_answer = None
        self._current_language = None

    def __repr__(self):
        return f"Q: {self.question}\nA: {self.answer}\nSn: {self.state_name}\nS: {self.state}"
    
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
                ".c": 0,
                "cpp": 0,
                "c++": 0,
                "rust": 0,
                "bash": 0,
                "terminal": 0
            }
            for i in range(idx-3, idx):
                for language in counter:
                    counter[language] += len(lines[i].split(language)) - 1
            if sum(counter.values()) == 0:
                # no language found, use default
                self._current_language = self.default_language
            else:
                self._current_language = max(counter, key=counter.get)
            # fix ".c" quirkkkk
            if self._current_language == '.c':
                self._current_language = 'c'
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
                stripped_block = '\n'.join(current_block).strip() + '\n'
                if stripped_block:
                    code_blocks.append((current_language, stripped_block))
                current_language = language
                current_block = [line]
        stripped_block = '\n'.join(current_block).strip() + '\n'
        if stripped_block:
            code_blocks.append((current_language, stripped_block))
        return code_blocks
    
    def highlight(self):
        ''' return the LLM answer, with syntax highlighting for code blocks
        '''
        if self._highlighted_answer is not None:
            return self._highlighted_answer
        lines = self.answer.splitlines()
        highlighted_lines = []
        for idx, line in enumerate(lines):
            language, code_block_delimiter = self.get_most_likely_language(lines, idx)
            if language and not code_block_delimiter:
                highlighted_lines.append(self._highlight(line, language))
            else:
                highlighted_lines.append(line + '\n')
        self._highlighted_answer = ''.join(highlighted_lines)
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
$ find . -name '*.py' | xargs grep 'def' | ./aish.py update-function aish.py test1
```
    '''
    def test_highlight():
        answer = LLMAnswer("", answer1, {}, None)
        # assert that the first code block is python
        print(answer.highlight())

    def test_get_code_blocks():
        answer = LLMAnswer("", answer1, {}, None)
        pp(answer.get_code_blocks())

    #test_highlight()
    test_get_code_blocks()
    sys.exit()


class Test_LLMAnswer(unittest.TestCase):
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
$ find . -name '*.py' | xargs grep 'def' | ./aish.py update-function aish.py test1
``` '''

    answer3 = '''
Here's an example implementation that fulfills the instructions you provided:

```
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

void create_temp_file() {
  char filename[] = "temp.txt";
  int fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0666);
  if (fd == -1) {
    perror("open");
    exit(1);
  }

  pid_t pid = getpid();
  uid_t uid = getuid();
  gid_t gid = getgid();

  // Write current pid, user, and other useful data to file
  dprintf(fd, "PID: %d\n", pid);
  dprintf(fd, "UID: %d\n", uid);
  dprintf(fd, "GID: %d\n", gid);

  close(fd);
}

int main() {
  printf("Hello, world!\n");
  create_temp_file();
  return 0;
}
```

This implementation creates a new temporary file in the current directory, writes the current process ID, user ID, and group ID to the file, and then closes it. You can modify the filename and file permissions as needed.
'''

    def test_get_code_blocks(self):
        answer = LLMAnswer("", self.answer1, {}, None, default_language="brainfuck")
        code_blocks = answer.get_code_blocks()
        self.assertEqual(len(code_blocks), 3)
        self.assertEqual(code_blocks[0][0], None)
        self.assertEqual(code_blocks[1][0], "python")
        self.assertEqual(code_blocks[2][0], None)
        self.assertEqual(code_blocks[0][1], 'Hey look at this dope python code I wrote:\n')
        self.assertEqual(code_blocks[1][1], 'def test1():\n    print("Hello1")\n')
        self.assertEqual(code_blocks[2][1], 'nice huh?\n')

    def test_get_code_blocks2(self):
        answer = LLMAnswer("", self.answer2, {}, None, default_language="malbolge")
        code_blocks = answer.get_code_blocks()
        self.assertEqual(len(code_blocks), 4)
        self.assertEqual(code_blocks[0][0], None)
        self.assertEqual(code_blocks[1][0], 'python')
        self.assertEqual(code_blocks[2][0], None)
        self.assertEqual(code_blocks[3][0], 'terminal')
        self.assertEqual(code_blocks[0][1], 'Hey look at this dope python code I wrote:\n')
        self.assertEqual(code_blocks[1][1], 'def test1():\n    print("Hello1")\n')
        self.assertEqual(code_blocks[2][1], 'nice huh? here we to some terminal stuff and in the terminal we invoke a python script\nsince we love\nlook here\n')
        self.assertEqual(code_blocks[3][1], '$ find . -name \'*.py\' | xargs grep \'def\' | ./aish.py update-function aish.py test1\n')
    
    def test_get_code_blocks3(self):
        answer = LLMAnswer("", self.answer3, {}, None, default_language="c")
        code_blocks = answer.get_code_blocks()
        self.assertEqual(len(code_blocks), 3)
        self.assertEqual(code_blocks[0][0], None)
        self.assertEqual(code_blocks[1][0], "c")
        self.assertEqual(code_blocks[2][0], None)


class LLM(object):

    state_dir = Path.home() / ".ai"
    state_name = ''

    def __init__(self, state_name=''):
        self.state_dir.mkdir(exist_ok=True)
        self.state_name = state_name or "state-{}".format(int(time.time()))
        self.state_path = self.state_dir / self.state_name
        if not self.state_path.exists():
            with open(self.state_path, 'w') as f:
                json.dump([], f)

    def use_state(self, state_name):
        self.state_name = state_name

    system_role_desc = 'You are a helpful assistant.'

    def set_system_role(self, desc):
        self.system_role_desc = desc

    def get_system_role(self):
        return self.system_role_desc

    def list_states(self):
        return [str(p.name) for p in self.state_dir.glob('*') if p.is_file()]

    def load_state(self, state_name=None):
        if state_name is None:
            state_name = self.state_name
        state_file = self.state_dir / state_name
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        else:
            return []

    def update_state(self, state):
        state_file = self.state_dir / self.state_name
        with open(state_file, 'w') as f:
            json.dump(state, f)

    def clear_state(self):
        state_file = self.state_dir / self.state_name
        if state_file.exists():
            os.remove(state_file)

    def load_model(self):
        model_file = Path.home() / ".aimodel"
        if model_file.exists():
            with open(model_file, 'r') as f:
                return f.read().strip()
        else:
            return "gpt-3.5-turbo"

    def ask(self, question, default_language=None):
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

        debug("Query: ", json.dumps(messages, indent=4))
        debug("Response: ", json.dumps(response, indent=4))

        answer = response['choices'][0].message.content
        state.append({"role": "user", "content": question})
        state.append({"role": "assistant", "content": answer})
        self.update_state(state)
        return LLMAnswer(question, answer, state, self.state_name, default_language=default_language)


class InteractiveShell(cmd.Cmd):
    def __init__(self):
        super().__init__()

    def parseline(self, line):
        self.last_command = line
        return super().parseline(line)

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
    
    
    #
    # run_autofix
    #
    def do_run_autofix(self, arg):
        # invoke python file, capture output and if exception is thrown,
        # ask LLM for a fix
        if isinstance(arg, str):
            args = self.parser.parse_args(f'run_autofix {arg}'.split())
        else:
            args = arg
        # execute using subprocess
        print(args.executable, args.arguments)
        p = subprocess.run(["python3", args.executable] + args.arguments, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        output = p.stdout.decode('utf-8')
        print(output)
        if p.returncode != 0:
            # exception was thrown
            # ask LLM for a fix
            question = f"Fix the following error in file `{args.executable}`: \n\n{output}"
            answer = LLM().ask(question, default_language='python')
            print(answer.highlight())
            while True:
                choice = input("Accept changes? [y/n/diff/show/<new_instruction>] ")
                if choice.lower() == 'y':
                    new_func_source = self.get_language_specific_code_block(answer, 'python')
                    with open(args.executable, 'w') as f:
                        f.write(new_func_source)
                    print(f"Updated file `{args.executable}`")
                    break
                elif choice.lower() == 'n':
                    break
                elif choice.lower() == 'diff':
                    code_block = self.get_language_specific_code_block(answer, 'python')
                    display_diff(output, code_block)
                elif choice.lower() == 'show':
                    print(answer.highlight())
                else:
                    new_instruction = choice
                    answer = LLM(state_name=answer.state_name).ask(new_instruction)
                    print(answer.highlight())
                

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter, PathCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory


class PythonFunctionCompleter(Completer):
    def __init__(self, filename):
        self.filename = filename
        self.function_names = self.get_function_names(filename)
        self.completer = WordCompleter(self.function_names, ignore_case=True)

    def get_completions(self, document: Document, complete_event):
        for completion in self.completer.get_completions(document, complete_event):
            yield completion

    @staticmethod
    def get_function_names(filename):
        with open(filename) as f:
            source = f.read()
        tree = ast.parse(source)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


class PythonClassCompleter(Completer):
    def __init__(self, filename):
        self.filename = filename
        self.class_names = self.get_class_names(filename)
        self.completer = WordCompleter(self.class_names, ignore_case=True)
    
    def get_completions(self, document: Document, complete_event):
        for completion in self.completer.get_completions(document, complete_event):
            yield completion

    @staticmethod
    def get_class_names(filename):
        with open(filename) as f:
            source = f.read()
        tree = ast.parse(source)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class CustomCompleter(Completer):
    commands = []
    command_names = []
    #command_completer = WordCompleter(commands, ignore_case=True)
    path_completer = PathCompleter()
    function_completer = None
    class_completer = None

    @classmethod
    def register_command(cls, command):
        cls.commands.append(command)
        cls.command_names.append(command.command_name)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.command_completer = WordCompleter(self.command_names, ignore_case=True)

    def get_completions(self, document: Document, complete_event):
        words = document.text.split(' ')

        # Complete the command
        if len(words) == 1:
            for completion in self.command_completer.get_completions(document, complete_event):
                yield completion

        elif len(words) >= 2:
            if words[0] == 'update-file':
                # Complete the file path
                if len(words) == 2:
                    for completion in self.path_completer.get_completions(Document(words[1]), complete_event):
                        yield completion

            elif words[0] == 'update-function':
                # Complete the file path
                if len(words) == 2:
                    for completion in self.path_completer.get_completions(Document(words[1]), complete_event):
                        yield completion
                # Complete the function name
                elif len(words) == 3 and words[1].endswith('.py'):
                    if self.function_completer is None or self.function_completer.filename != words[1]:
                        self.function_completer = PythonFunctionCompleter(words[1])
                    for completion in self.function_completer.get_completions(Document(words[2]), complete_event):
                        yield completion

            elif words[0] == 'update-class':
                # Complete the file path
                if len(words) == 2:
                    for completion in self.path_completer.get_completions(Document(words[1]), complete_event):
                        yield completion
                # Complete the class name
                elif len(words) == 3 and words[1].endswith('.py'):
                    if self.class_completer is None or self.class_completer.filename != words[1]:
                        self.class_completer = PythonClassCompleter(words[1])
                    for completion in self.class_completer.get_completions(Document(words[2]), complete_event):
                        yield completion

            elif words[0] == 'help':
                for completion in self.command_completer.get_completions(Document(' '.join(words[1:])), complete_event):
                    yield completion


def help(args):
    help_texts = {
        'update-file': 'update-file <file>: Updates the specified file',
        'update-function': 'update-function <file> <function>: Updates the specified function in the file',
        'add-numbers': 'add-numbers <nums>: Adds the specified numbers',
        'help': 'help <command>: Shows help text for the specified command'
    }
    print(help_texts.get(args[0], 'Unknown command'))


class Command(object):
    command_name=''
    help_text=''

    def __init__(self, subparsers):
        self.parser = subparsers.add_parser(self.command_name)
        self.parser.set_defaults(func=self.func)
        self._init_arguments()
        CustomCompleter.register_command(self)
    
    def _init_arguments(self):
        pass
    
    def func(self, args):
        print(f"{self.command_name} executed with arguments: {args}")


class AskCommand(Command):
    command_name = 'ask'
    help_text = 'ask <question>: Ask LLM a question'

    def _init_arguments(self):
        self.parser.add_argument('question', nargs='+', help='Update instruction to LLM')

    def func(self, args):
        question = ' '.join(args.question)
        answer = LLM().ask(question)
        print(answer)


class SummarizeFileCommand(Command):
    command_name = 'summarize-file'
    help_text = 'summarize-file <file_path>: Ask LLM to summarize the contents of `file_path`'

    def _init_arguments(self):
        self.parser.add_argument('file_path', help='File to be summarized')
    
    def func(self, args):
        with open(args.file_path) as f:
            contents = f.read()
        question = f"Summarize the file `{args.file_path}`, file contents pasted below:\n\n{contents}"
        answer = LLM().ask(question)
        print(answer)


class _LanguageHelpers(object):
    def _get_file_language(self, filename):
        ''' return the language of the file, based on the file extension
        '''
        if filename.endswith('.py'):
            return 'python'
        elif filename.endswith('.go'):
            return 'golang'
        elif filename.endswith('.c'):
            return 'c'
        elif filename.endswith('.cpp'):
            return 'cpp'
        elif filename.endswith('.rs'):
            return 'rust'
        elif filename.endswith('.sh'):
            return 'bash'
        else:
            return None 
    
    def get_language_specific_code_block(self, answer, language):
        code_blocks = answer.get_code_blocks()
        matching_block = ''
        count = 0
        for lang, code_block in code_blocks:
            if lang == language:
                matching_block = code_block
                count += 1
        if count > 1:
            raise Exception(f"Found multiple code blocks for language {language}")
        elif count == 0:
            lang, code_block = code_blocks[0]
            # try parse code_block as python code
            try:
                ast.parse(code_block)
            except SyntaxError:
                raise Exception(f"Found no code blocks for language {language}. Parsing the first code block as {lang} failed: {code_block}")
            else:
                matching_block = code_block
        return matching_block


class _PythonCommandHelpers(object):
    def _update_node(self, command_name, file_path, node_type, node_name, instruction, yes=False):
        instruction = ' '.join(instruction)
        file_language = self._get_file_language(file_path)
        if file_language != 'python':
            raise Exception(f"{command_name} only supports python at the moment, not {file_language}")
        pymod = PyMod(file_path)
        node_source = pymod.get_source_of_node(node_name, node_type)
        question = f"Suggest how to update the {node_type} `{node_name}` in file `{file_path}`, "
        question += f" reply with code suggestions contained within triple backticks, "
        question += f" according to the following instructions: `{instruction}`"
        question += f"\n\n{node_source}" 
        answer = LLM().ask(question, default_language=file_language)
        print(answer.highlight())
        while True:
            choice = ''
            if yes:
                choice = 'y'
            else:
                choice = input(f"[{self.command_name}] Accept changes? [y/n/diff/show/<new_instruction>] ")
            if choice.strip() == '':
                continue
            elif choice.lower() == 'y':
                new_node_source = self.get_language_specific_code_block(answer, file_language)
                pymod.replace_node(node_type, node_name, new_node_source)
                print(f"Updated function `{node_name}` in file `{file_path}`")
                break
            elif choice.lower() == 'n':
                break
            elif choice.lower() == 'diff':
                code_block = self.get_language_specific_code_block(answer, file_language)
                display_diff(node_source, code_block)
            elif choice.lower() == 'show':
                print(answer.highlight())
            else:
                new_instruction = choice
                answer = LLM(state_name=answer.state_name).ask(new_instruction)
                print(answer.highlight())


class UpdateFileCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    # TODO make update-file less language-centric to support other file types
    # TODO decide how multiple files shall be specified on cmdline. Single argument instruction?
    command_name = 'update-file'
    help_text = 'update-file <file_path> <instruction...>: Updates the specified file'

    def _init_arguments(self):
        self.parser.add_argument('-y', '--yes', action='store_true', help='Accept all changes without prompting')
        self.parser.add_argument('file_path', help='File to be updated')
        self.parser.add_argument('instruction', nargs='+', help='Update instruction to LLM')
    
    def func(self, args):
        debug(f"{self.command_name} executed with arguments: {args}")
        instruction = ' '.join(args.instruction)
        with open(args.file_path) as f:
            contents = f.read()
        question = f"Suggest how to update the file `{args.file_path}`, "
        question += f" code suggestions contained within triple backticks, "
        question += f" according to the following instructions: `{instruction}`"
        question += f"\n\n{contents}" 
        file_language = self._get_file_language(args.file_path)
        answer = LLM().ask(question, default_language=file_language)
        print(answer.highlight())
        while True:
            choice = None
            if args.yes:
                choice = 'y'
            else:
                choice = input("Accept changes? [y/n/diff/show/<new_instruction>] ")
            
            if choice.strip() == '':
                continue
            elif choice.lower() == 'y':
                new_content = self.get_language_specific_code_block(answer, file_language)
                backup_file_path = f"{args.file_path}.bak.{datetime.now().strftime('%H%M%S')}"
                with open(backup_file_path, 'w') as f:
                    f.write(contents)
                with open(args.file_path, 'w') as f:
                    f.write(new_content)
                print(f"Updated file `{args.file_path}`")
                break
            elif choice.lower() == 'n':
                break
            elif choice.lower() == 'diff':
                code_block = self.get_language_specific_code_block(answer, file_language)
                display_diff(contents, code_block)
            elif choice.lower() == 'show':
                print(answer.highlight())
            else:
                new_instruction = choice
                answer = LLM(state_name=answer.state_name).ask(new_instruction)
                print(answer.highlight())


class UpdateFilesCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    # TODO make update-file less language-centric to support other file types
    # TODO decide how multiple files shall be specified on cmdline. Single argument instruction?
    # WORKS on cmdline but not in interactive shell
    command_name = 'update-files'
    help_text = 'update-files [-y] "instruction" files...: Update files according to instruction:'

    def _init_arguments(self):
        self.parser.add_argument('-y', '--yes', action='store_true', help='Accept all changes without prompting')
        self.parser.add_argument('instruction', help='File update instruction to LLM')
        self.parser.add_argument('files', nargs='+', help='Files to be updated')
    
    def func(self, args):
        debug(f"{self.command_name} executed with arguments: {args}")
        instruction = args.instruction
        for file_path in args.files:
            with open(file_path) as f:
                contents = f.read()
            question = f"Suggest how to update the file `{file_path}`, "
            question += f" code suggestions contained within triple backticks, "
            question += f" according to the following instructions: `{instruction}`"
            question += f"\n\n{contents}" 
            file_language = self._get_file_language(file_path)
            answer = LLM().ask(question, default_language=file_language)
            print(answer.highlight())
            while True:
                choice = None
                if args.yes:
                    choice = 'y'
                else:
                    choice = input("Accept changes? [y/n/diff/show/<new_instruction>] ")
                
                if choice.strip() == '':
                    continue
                elif choice.lower() == 'y':
                    new_content = self.get_language_specific_code_block(answer, file_language)
                    backup_file_path = f"{file_path}.bak.{datetime.now().strftime('%H%M%S')}"
                    with open(backup_file_path, 'w') as f:
                        f.write(contents)
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    print(f"Updated file `{file_path}`")
                    break
                elif choice.lower() == 'n':
                    break
                elif choice.lower() == 'diff':
                    code_block = self.get_language_specific_code_block(answer, file_language)
                    display_diff(contents, code_block)
                elif choice.lower() == 'show':
                    print(answer.highlight())
                else:
                    new_instruction = choice
                    answer = LLM(state_name=answer.state_name).ask(new_instruction)
                    print(answer.highlight())


class UpdateFunctionCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    command_name = 'update-function'
    help_text = 'update-function <file_path> <func_name>: Updates the specified function in the file'

    def _init_arguments(self):
        self.parser.add_argument('-y', '--yes', action='store_true', help='Accept all changes without prompting')
        self.parser.add_argument('file_path', help='File containing function')
        self.parser.add_argument('func_name', help='Function to be updated')
        self.parser.add_argument('instruction', nargs='+', help='Update instruction to LLM')
    
    def func(self, args):
        return self._update_node(self.command_name, args.file_path, ast.FunctionDef, args.func_name,
                                 args.instruction, yes=args.yes)


class UpdateClassCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    command_name = 'update-class'
    help_text = 'update-class <file_path> <class_name>: Updates the specified class in the file'

    def _init_arguments(self):
        self.parser.add_argument('-y', '--yes', action='store_true', help='Accept all changes without prompting')
        self.parser.add_argument('file_path', help='File containing class')
        self.parser.add_argument('class_name', help='Class to be updated')
        self.parser.add_argument('instruction', nargs='+', help='Update instruction to LLM')
    
    def func(self, args):
        return self._update_node(self.command_name, args.file_path, ast.ClassDef, args.class_name,
                                 args.instruction, yes=args.yes)

        
class ListFunctionsCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    command_name = 'list-functions'
    help_text = 'list-functions <file_path>: Lists the functions in the specified file'

    def _init_arguments(self):
        self.parser.add_argument('file_path', help='Python file containing functions')
    
    def func(self, args):
        with open(args.file_path) as f:
            source = f.read()
        tree = ast.parse(source)
        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print('\n'.join(function_names))


class ListClassesCommand(Command, _LanguageHelpers, _PythonCommandHelpers):
    command_name = 'list-classes'
    help_text = 'list-classes <file_path>: Lists the classes in the specified file'

    def _init_arguments(self):
        self.parser.add_argument('file_path', help='Python file containing classes')
    
    def func(self, args):
        with open(args.file_path) as f:
            source = f.read()
        tree = ast.parse(source)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        print('\n'.join(class_names))


class AddNumbersCommand(Command):
    command_name = 'add-numbers'
    help_text = 'add-numbers <nums>: Adds the specified numbers'

    def _init_arguments(self):
        self.parser.add_argument('nums', nargs='+', type=int, help='Numbers to be added')
    
    def func(self, args):
        print(f"{self.command_name} executed with arguments: {args}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--debug', action='store_true', help='Enable debugging')
    subparsers = parser.add_subparsers()

    ask = AskCommand(subparsers)
    summarize_file = SummarizeFileCommand(subparsers)
    update_file = UpdateFileCommand(subparsers)
    update_files = UpdateFilesCommand(subparsers)
    update_func = UpdateFunctionCommand(subparsers)
    update_class = UpdateClassCommand(subparsers)
    list_functions = ListFunctionsCommand(subparsers)
    list_classes = ListClassesCommand(subparsers)
    add_numbers = AddNumbersCommand(subparsers)

    if len(sys.argv) != 1:  # No arguments, switch to interactive mode
        args = parser.parse_args()
        if args.debug:
            logging.basicConfig(level=logging.DEBUG)
        return args.func(args)
    # 
    # interactive mode
    #
    session = PromptSession(completer=CustomCompleter(), history=FileHistory(os.path.expanduser('~/.aish_history')))

    #
    # monkeypatch argparse.error so that it doesn't exit on invalid input
    #
    def argparse_error(self, message):
        self.print_usage()
    def argparse_exit(self, status=0, message=None):
        pass

    argparse.ArgumentParser.exit = argparse_exit

    while True:
        try:
            text = session.prompt("[gpt-3.5-turbo]:AISh >>> ", auto_suggest=AutoSuggestFromHistory())
        except KeyboardInterrupt:
            continue  # Ctrl+C pressed, let's ask for input again
        except EOFError:
            break  # Ctrl+D pressed, let's exit
        if text.strip() == '':
            continue

        args, unknown = parser.parse_known_args(text.split())

        if unknown:
            print("Unknown command or arguments: " + ' '.join(unknown))
        else:
            if hasattr(args, 'func'):
                args.func(args)
            else:
                print(f"Unknown command: {text}")


if __name__ == "__main__":
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        unittest.main()
        sys.exit()
    sys.exit(main())


#import subprocess
#import pexpect
#
#if 0: 
#    class Test_InteractiveShell(unittest.TestCase):
#        def test_sum_command_line(self):
#            output = subprocess.check_output(["python3", "aish.py", "sum", "1", "2"])
#            self.assertEqual(output.strip(), b'3')
#
#        def test_greet_command_line(self):
#            output = subprocess.check_output(["python3", "aish.py", "greet", "Alice"])
#            self.assertEqual(output.strip(), b'Hello, Alice!')
#
#        def test_sum_interactive_shell(self):
#            child = pexpect.spawn('python3 aish.py')
#            child.sendline('sum 1 2')
#            child.expect('3')
#            child.sendline('quit')
#
#        def test_greet_interactive_shell(self):
#            child = pexpect.spawn('python3 aish.py')
#            child.sendline('greet Alice')
#            child.expect('Hello, Alice!')
#            child.sendline('quit')
#
#if __name__ == '__main__':
#    import sys
#    if '--test' in sys.argv:
#        sys.argv.remove('--test')
#        unittest.main()
#        sys.exit()
#
#    if len(sys.argv) > 1:
#        # Command line mode
#        shell = InteractiveShell()
#        try:
#            shell.parser.parse_args(sys.argv[1:]).func(shell.parser.parse_args(sys.argv[1:]))
#        except:
#            extype, value, tb = sys.exc_info()
#            traceback.print_exc()
#            pdb.post_mortem(tb)
#    else:
#
#        # Interactive mode
#        try:
#            InteractiveShell().cmdloop()
#        except:
#            extype, value, tb = sys.exc_info()
#            traceback.print_exc()
#            pdb.post_mortem(tb)
####################################################
