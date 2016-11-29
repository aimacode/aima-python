# The GUI was modified from http://www.tkdocs.com/tutorial/firstexample.html. The original material
# is copyrighted by Mark Roseman. The license can be found at https://creativecommons.org/licenses/by-nc-sa/2.5/ca/.
# See the License for the specific language governing permissions and limitations under the License.
from tkinter import *
from tkinter import ttk
from submissions.LaMartina import GoogleCSPSolver

#GoogleCSPSolver.main(problem_str="SEND+MORE=MONEY", base = 10)

# def calculate(*args):
#     try:
#         value = float(feet.get())
#         meters.set((0.3048 * value * 10000.0 + 0.5) / 10000.0)
#     except ValueError:
#         pass

def solve(*args):
    try:
        setInputString(topWord,botWord,answer)
        if inputString.get() == "":
            solution.set("Please input 3 Numbers!")
        else:
            solution.set("")
            GoogleCSPSolver.main(problem_str=inputString.get(), base=10)
            stringprint = GoogleCSPSolver.Stringtoprint
            solution.set(stringprint)
        # value = float(topWord.get())
        # solution.set((0.3048 * value * 10000.0 + 0.5) / 10000.0)
    except ValueError:
        pass
def setInputString(top, bottom, sum):
    try:
        if top.get() == "" or bottom.get() == "" or sum.get() == "":
            inputString.set("")
        else:
            inputString.set(top.get().capitalize() + "+" + bottom.get().capitalize() + "=" + sum.get().capitalize())

    except ValueError:
        pass


root = Tk()
root.title("Alphametic Solver")
#root.title("Feet to Meters")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# feet = StringVar()
# meters = StringVar()
topWord = StringVar()
botWord = StringVar()
answer = StringVar()
solution = StringVar()
inputString = StringVar()

# topWord.set("")
# botWord.set("")
# answer.set("")
# solution.set("")
#inputString.set("")

# topWord = ""
# botWord = ""
# answer = ""
# solution = ""

topWord_entry = ttk.Entry(mainframe, width=7, textvariable=topWord)
topWord_entry.grid(column=2, row=2, sticky=(W, E))
botWord_entry = ttk.Entry(mainframe, width=7, textvariable=botWord)
botWord_entry.grid(column=2, row=3, sticky=(W, E))
answer_entry = ttk.Entry(mainframe, width=7, textvariable=answer)
answer_entry.grid(column=2, row=5, sticky=(W, E))

#ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Solve", command=solve).grid(column=3, row=7, sticky=W)
ttk.Label(mainframe, textvariable=solution).grid(column=2, row=6, sticky=(W, E))

ttk.Label(mainframe, text="Insert a two Word Alphametric Problem!").grid(column=2, row=1, sticky=W)
ttk.Label(mainframe, text="Top Word").grid(column=3, row=2, sticky=W)
ttk.Label(mainframe, text="+").grid(column=1, row=3, sticky=E)
ttk.Label(mainframe, text="Bottom Word").grid(column=3, row=3, sticky=W)
ttk.Label(mainframe, text ="___________________________________________").grid(column=2, row=4, sticky=(W,E))
ttk.Label(mainframe, text="Sum").grid(column=3, row=5, sticky=W)
ttk.Label(mainframe, text="Solution").grid(column=3, row=6, sticky=W)
#ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)
# ttk.Label(mainframe, text="top").grid(column=3, row=1, sticky=W)
# ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
# ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

#inputString = setInputString(topWord,botWord,answer)
#solution = solve()

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

topWord_entry.focus()
root.bind('<Return>', solve)

root.mainloop()