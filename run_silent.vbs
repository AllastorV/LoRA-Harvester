' LoRA-Harvester Silent Launcher
' Launches the app without showing a CMD/console window
Option Explicit

Dim oShell, sDir, sCmd

Set oShell = CreateObject("WScript.Shell")

' Get the directory of this script
sDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))

' Set working directory
oShell.CurrentDirectory = sDir

' Build command: try pythonw first (truly windowless), fallback to python
Dim sPython
sPython = "pythonw"
On Error Resume Next
oShell.Run "pythonw --version", 0, True
If Err.Number <> 0 Then
    sPython = "python"
End If
On Error GoTo 0

sCmd = sPython & " """ & sDir & "main.py"""

' WindowStyle=0 = hidden, bWaitOnReturn=False = non-blocking
oShell.Run sCmd, 0, False

Set oShell = Nothing
