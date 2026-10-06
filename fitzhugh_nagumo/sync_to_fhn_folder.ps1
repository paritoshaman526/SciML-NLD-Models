# Synchronize all code, models, and artifacts to c:\Users\satya\Downloads\FitzHugh Nagumo
$source = "c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo"
$dest = "c:\Users\satya\Downloads\FitzHugh Nagumo"

Write-Host "Syncing from $source to $dest..."
Copy-Item -Path "$source\*" -Destination "$dest\" -Recurse -Force
Write-Host "Sync completed successfully."
