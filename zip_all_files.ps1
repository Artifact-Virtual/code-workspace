# PowerShell script to compress all files in the current directory and subdirectories into a zip file
$zipFile = "artifactvirtual_backup.zip"
$root = Get-Location
if (Test-Path $zipFile) { Remove-Item $zipFile }
Compress-Archive -Path "$root\*" -DestinationPath $zipFile -Force
Write-Host "All files compressed into $zipFile"
