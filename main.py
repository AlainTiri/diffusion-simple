# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import diffusion


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    model = diffusion.model()

    while True:
        prompt = input("Give me a prompt (or exit) : ")
        if prompt == "exit": break
        model.text2img(prompt)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
