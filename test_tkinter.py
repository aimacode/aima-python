from tkinter import *  
import PIL.Image
import PIL.ImageTk


im = PIL.Image.open("./img/fire.png")
photo = PIL.ImageTk.PhotoImage(im)

label = Label(root, image=photo)
label.image = photo  # keep a reference!
label.pack()

root.mainloop()