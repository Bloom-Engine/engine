# Put the DirectX Shader Compiler next to a Bloom binary.
#
#   powershell -File tools/fetch-dxc.ps1 -Dest C:\path\to\game
#
# Why this exists: wgpu's DX12 backend reports the adapter's shader model as
# min(device, compiler). Its fallback compiler, FXC, caps that at 5.1 - and
# hardware ray query needs 6.5. So without DXC, EXPERIMENTAL_RAY_QUERY is never
# exposed on DX12 on ANY GPU, Lumen silently drops to its software trace, and
# the only symptom is "ray_query=false" in the boot line.
#
# wgpu loads dxcompiler.dll by name, so the DLL must sit beside the binary (or
# on PATH). dxil.dll signs the compiled shaders; without it the driver rejects
# them. Both ship with the Windows SDK, so copy rather than download.
#
# If the SDK is absent, take them from
# https://github.com/microsoft/DirectXShaderCompiler/releases (v1.8.2502+).
# Missing DLLs are not fatal: wgpu falls back to FXC and you are back on the
# software GI path.
#
# ASCII only on purpose - PowerShell reads .ps1 as ANSI, and a stray em-dash
# is a parse error.
param(
    [Parameter(Mandatory = $true)][string]$Dest
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Dest)) { throw "destination not found: $Dest" }

$sdkBin = 'C:\Program Files (x86)\Windows Kits\10\bin'
if (-not (Test-Path $sdkBin)) { throw "Windows SDK not found at $sdkBin" }

# Newest SDK version that ships an x64 dxcompiler.dll.
$src = Get-ChildItem $sdkBin -Directory |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName 'x64' } |
    Where-Object { Test-Path (Join-Path $_ 'dxcompiler.dll') } |
    Select-Object -First 1

if (-not $src) { throw "no x64 dxcompiler.dll under $sdkBin" }

foreach ($dll in @('dxcompiler.dll', 'dxil.dll')) {
    $from = Join-Path $src $dll
    if (-not (Test-Path $from)) { throw "missing $dll in $src" }
    Copy-Item $from -Destination $Dest -Force
    Write-Host "copied $dll -> $Dest"
}

Write-Host "DXC in place. Boot line should now report ray_query=true on an RT-capable GPU."
