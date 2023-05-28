#!/usr/bin/env python3
#
# TODO:
# - add state size to prompt
# - add all interactive commands to argparse
#
import openai
import os
import sys
import json
import argparse
from pathlib import Path
import readline
import pprint

from pygments import highlight
from pygments.lexers import PythonLexer, GoLexer, CLexer, CppLexer, RustLexer, BashLexer
from pygments.formatters import TerminalFormatter


openai.api_key = os.getenv('OPENAI_API_KEY')


COMMANDS = ['clear-state', 'show-state', 'use-state', 'list-state', 'save-state',
            'update-file', 'update-file-retry',
            'summarize-file', 'explain-file',
            'set-system-role', 'get-system-role', 'debug', 'help', 'quit']


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


def extract_python_function(filename, func_name):
    # extract source code of function from filename
    with open(filename, 'r') as f:
        content = f.read()
    # find the function definition
    func_def = 'def {}'.format(func_name)
    func_start = content.find(func_def)
    if func_start == -1:
        raise Exception(f"Could not find function definition for {func_name}")
    # find the end of the function
    func_end = content.find('\n\n', func_start)
    if func_end == -1:
        raise Exception(f"Could not find end of function definition for {func_name}")
    # extract the function
    func = content[func_start:func_end]
    return func, func_def, func_start, func_end

    
def complete(text, state):
    for cmd in COMMANDS:
        if cmd.startswith(text):
            if not state:
                return cmd
            else:
                state -= 1

readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

ai_dir = Path.home() / ".ai"
curr_state = "curr"
debug = False


def use_state(state_name):
    global curr_state
    curr_state = state_name


system_role_desc = 'You are a helpful assistant.'

def set_system_role(desc):
    global system_role_desc
    system_role_desc = desc

def get_system_role():
    global system_role_desc
    return system_role_desc

def list_states():
    return [str(p.name) for p in ai_dir.glob('*') if p.is_file()]

def load_state(state_name=curr_state):
    state_file = ai_dir / state_name
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    else:
        return []

def save_state(save_name):
    curr_file = ai_dir / curr_state
    save_file = ai_dir / save_name
    if save_file.exists():
        raise Exception("State already exists")
    print(f"Saving state to {save_file} from {curr_file}")
    with open(save_file, 'w') as fw:
        with open(curr_file, 'r') as fr:
            fw.write(fr.read())

def update_state(state):
    state_file = ai_dir / curr_state
    with open(state_file, 'w') as f:
        json.dump(state, f)

def clear_state():
    state_file = ai_dir / curr_state
    if state_file.exists():
        os.remove(state_file)

def load_model():
    model_file = Path.home() / ".aimodel"
    if model_file.exists():
        with open(model_file, 'r') as f:
            return f.read().strip()
    else:
        return "gpt-3.5-turbo"

def extract_single_block_of_python_code_from_GPT_answer(answer):
    code_block_delimiter = '```'
    #
    # sometimes code block delimiter is missing
    #
    if not code_block_delimiter in answer:
        return answer
    if answer.count(code_block_delimiter) != 2:
        raise Exception("Expected 0 or 2 code block delimiters, found", answer.count(code_block_delimiter))
    code_block_start = answer.find(code_block_delimiter)
    if code_block_start == -1:
        return answer
    code_block_end = answer.find(code_block_delimiter, code_block_start + len(code_block_delimiter))
    if code_block_end == -1:
        raise Exception("Could not find code block end")
    code_block = answer[code_block_start + len(code_block_delimiter):code_block_end]
    return code_block


def update_file(filename, instruction):
    func_name = func_def = func_start = func_end = None
    is_python_file = False
    if ':' in filename:
        filename, func_name = filename.split(':')
        #
        # Only Python support atm.
        #
        if filename.endswith('.py'):
            is_python_file = True
        assert is_python_file
        content, func_def, func_start, func_end = extract_python_function(filename, func_name)
        #content = content.replace(f'def {func_name}', 'def func')
        msg = f"Update the function {func_name} in {filename}, file contents pasted below the two newlines. "
        msg += f"Return nothing but the updated function definition, "
        msg += f"commented, according the following instruction: {instruction}"
        msg += '\n\n'
        msg += content
    else:
        if filename.endswith('.py'):
            is_python_file = True
        with open(filename, 'r') as f:
            content = f.read()
        msg = f"Update the file {filename}, file contents pasted below the two newlines. "
        msg += f"Return nothing but the updated file contents, "
        msg += f"commented, according the following instruction: {instruction}"
        msg += '\n\n'
        msg += content

    lexer = None
    if filename.endswith('.py'):
        lexer = PythonLexer()
    if filename.endswith('.go'):
        lexer = GoLexer()
    if filename.endswith('.c'):
        lexer = CLexer()
    if filename.endswith('.cpp') or filename.endswith('.cc'):
        lexer = CppLexer()
    if filename.endswith('.sh') or filename.endswith('.bash'):
        lexer = BashLexer()


    answer = ask_gpt3(msg)
    #
    # syntax highlighting
    #
    hl_answer = answer
    if not lexer is None:
        hl_answer = highlight(answer, lexer, TerminalFormatter())
    
    if is_python_file:
        #
        # extract function definition from answer
        #
        answer = extract_single_block_of_python_code_from_GPT_answer(answer)
        #
        # Keep entire response in highligted response in interactive mode
        #
        #hl_answer = highlight(answer, PythonLexer(), TerminalFormatter())

    print(f"{hl_answer}")

    while True:
        accept = input("Accept? (y/n/diff/show/<new instruction>): ")
        if accept.lower() == 'diff':
            display_diff(content, answer)
        elif accept.lower() == 'show':
            print(f"{hl_answer}")
        elif accept.lower() not in ('y', 'n'):
            instruction = accept
            answer = ask_gpt3(msg)
            #
            # syntax highlighting
            #
            hl_answer = answer
            if not lexer is None:
                hl_answer = highlight(answer, lexer, TerminalFormatter())

            print(f"{hl_answer}")
        else:
            break

    if accept.lower() == 'y':
        #
        # save backup of original file
        #
        backup_filename = filename + '.bak.' + str(int(os.path.getmtime(filename)))
        with open(backup_filename, 'w') as f:
            f.write(content)

        if not func_name is None:
            #
            # replace function definition
            #
            with open(filename, 'w') as f:
                f.write(content[:func_start])
                f.write(answer)
                f.write(content[func_end:])
        else:
            #
            # update file
            #
            with open(filename, 'w') as f:
                f.write(answer)
        
        print("Update accepted. Backup saved to", backup_filename)

    elif accept.lower() == 'n':
        print("Update rejected.")

    else:
        raise Exception('should not happen')


def explain_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    msg = f"Explain the file {filename}, file contents pasted below the two newlines. "
    msg += '\n\n'
    msg += content
    answer = ask_gpt3(msg)
    print(f"Explanation:\n\n{answer}")


def summarize_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    msg = f"Summarize the file {filename}, file contents pasted below the two newlines. "
    msg += '\n\n'
    msg += content
    answer = ask_gpt3(msg)
    print(f"Summary:\n\n{answer}")
    

def ask_gpt3(question):
    global debug
    state = load_state()
    model = load_model()

    messages = [{"role": "system", "content": system_role_desc}]
    for h in state:
        messages.append(h)
    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    if debug:
        print("Query: ", json.dumps(messages, indent=4))
        print("Response: ", json.dumps(response, indent=4))

    answer = response['choices'][0].message.content
    state.append({"role": "user", "content": question})
    state.append({"role": "assistant", "content": answer})
    update_state(state)

    return answer


def main():
    global debug
    parser = argparse.ArgumentParser()
    parser.add_argument('questions', nargs='*', help='Your questions to the assistant')
    parser.add_argument('--debug', action='store_true', help='Print the JSON query and response')
    parser.add_argument('--clear-state', action='store_true', help='Clear the state')
    parser.add_argument('--show-state', action='store_true', help='Show the state')
    parser.add_argument('--use-state', help='Use the specified state')
    parser.add_argument('--save-state', help='Save current state to a specific name')
    parser.add_argument('--list-state', action='store_true', help='List all available states')
    args = parser.parse_args()

    if args.debug:
        debug = True
    if args.clear_state:
        clear_state()
    elif args.show_state:
        pprint.pp(load_state())
    elif args.use_state:
        load_state(args.use_state)
    elif args.save_state:
        save_state(args.save_state)
    elif args.list_state:
        pprint.pp(list_states())
    else:
        model = load_model()
        if not args.questions:
            while True:
                question = input(f"{model} state[{curr_state}] >> ")
                #
                # state functions
                #

                # clear-state
                if question.lower() == 'clear-state':
                    clear_state()
                    print("History cleared.")
                # show-state
                elif question.lower() == 'show-state':
                    pprint.pp(load_state(curr_state))
                elif question.lower().startswith('show-state'):
                    state_name = question.split(' ', 1)[1]
                    pprint.pp(load_state(state_name))
                # use-state
                elif question.lower().startswith('use-state'):
                    state_name = question.split(' ', 1)[1]
                    use_state(state_name)
                # save-state
                elif question.lower().startswith('save-state'):
                    state_name = question.split(' ', 1)[1]
                    save_state(state_name)
                    print(f"Saved state '{state_name}'")
                # list-state
                elif question.lower() == 'list-state':
                    pprint.pp(list_states())
                
                #
                # file commands
                #

                # update-file
                elif question.startswith('update-file'):
                    #
                    # clears state to avoid huge resubmission of files
                    #
                    _, filename, instruction = question.split(' ', 2)
                    clear_state() # clear state to avoid resubmission of old files
                    update_file(filename, instruction)
                elif question.startswith('update-file-retry'):
                    #
                    # does NOT clear state
                    #
                    _, filename, instruction = question.split(' ', 2)
                    update_file(filename, instruction)
                elif question.lower().startswith('summarize-file'):
                    filename = question.split(' ', 1)[1]
                    summarize_file(filename)
                elif question.lower().startswith('explain-file'):
                    filename = question.split(' ', 1)[1]
                    explain_file(filename)
                
                #
                # system functions
                #

                # get-system-role
                elif question.lower() == 'get-system-role':
                    print(get_system_role())
                # set-system-role
                elif question.lower().startswith('set-system-role'):
                    desc = question.split(' ', 1)[1]
                    set_system_role(desc)
                    print(f"Set system role to '{desc}'")
                # debug
                elif question.lower() == 'debug':
                    debug = not(debug)
                    print('Debug mode: ', debug)
                else:
                    answer = ask_gpt3(question)
                    print("Answer: ", answer)
        else:
            for question in args.questions:
                answer = ask_gpt3(question)
                print(answer)

if __name__ == "__main__":
    main()

