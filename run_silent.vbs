' LoRA-Harvester Silent Launcher
' Launches the app without showing a CMD/console window
Option Explicit

Dim oShell, sDir, sRoot, sCmd, sPython

Set oShell = CreateObject("WScript.Shell")

' Script is at project root — use its own directory as working dir
sRoot = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
oShell.CurrentDirectory = sRoot

' Prefer venv pythonw (truly windowless), fall back to system python
Dim sVenvPyw
sVenvPyw = sRoot & "venv\Scripts\pythonw.exe"

Dim oFSO
Set oFSO = CreateObject("Scripting.FileSystemObject")

If oFSO.FileExists(sVenvPyw) Then
    sPython = """" & sVenvPyw & """"
Else
    ' Try system pythonw
    On Error Resume Next
    oShell.Run "pythonw --version", 0, True
    If Err.Number = 0 Then
        sPython = "pythonw"
    Else
        sPython = "python"
    End If
    On Error GoTo 0
End If

sCmd = sPython & " """ & sRoot & "main.py"""

' WindowStyle=0 = hidden window, bWaitOnReturn=False = non-blocking
oShell.Run sCmd, 0, False

Set oFSO = Nothing
Set oShell = Nothing
