$ErrorActionPreference = 'Stop'
$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder 'SMS AC Local Website.lnk'
$shellLink = New-Object -ComObject WScript.Shell
$shortcut = $shellLink.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
$shortcut.Arguments = '"' + (Join-Path $PSScriptRoot 'scripts\run_local_website.py') + '"'
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = 'SMS AC authenticated website and local trained model'
$shortcut.Save()
Write-Output "Starts after Windows login: $shortcutPath"
