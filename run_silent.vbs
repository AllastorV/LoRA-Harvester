Option Explicit
Dim shell, root, fso, exe
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName) & "\"
shell.CurrentDirectory = root
exe = root & "venv\Scripts\pythonw.exe"
If fso.FileExists(exe) Then
    shell.Run """" & exe & """ """ & root & "main.py""", 0, False
Else
    shell.Run """" & root & "install.bat""", 1, False
End If
