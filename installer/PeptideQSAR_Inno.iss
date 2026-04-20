#define MyAppName "Peptide QSAR Prediction Tool"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Ahmed G. Soliman"
#define MyAppExeName "PeptideQSAR.exe"

[Setup]
AppId={{9E8D3B34-7E8E-4B9D-9AE7-PEPTIDEQSAR01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\PeptideQSAR
DefaultGroupName=Peptide QSAR Tool
DisableProgramGroupPage=yes
OutputDir=C:\Users\PC\Documents\Peptide_QSAR_Tool\installer
OutputBaseFilename=PeptideQSAR_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

[Files]
Source: "C:\Users\PC\Documents\Peptide_QSAR_Tool\dist\PeptideQSAR.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\PC\Documents\Peptide_QSAR_Tool\dist\README_EXE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Peptide QSAR Tool"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall Peptide QSAR Tool"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Peptide QSAR Tool"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Peptide QSAR Tool"; Flags: nowait postinstall skipifsilent
