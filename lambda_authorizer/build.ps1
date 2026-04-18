# Setup script parameter
$OutputDir = "C:\Users\Lenovo\Desktop\voice-assistant\lambda_authorizer"
$ZipTarget = "C:\Users\Lenovo\Desktop\voice-assistant\authorizer.zip"

Write-Host "Installing PyJWT into package directory..."
pip install pyjwt -t $OutputDir

Write-Host "Creating deployment zip..."
# Delete the old zip if it exists
if (Test-Path $ZipTarget) { Remove-Item $ZipTarget }

# Compress the entire directory into a zip file
Compress-Archive -Path "$OutputDir\*" -DestinationPath $ZipTarget

Write-Host "Done! Your Lambda deployment package is ready at: $ZipTarget"
