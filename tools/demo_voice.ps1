# Synthesize the demo narration with a Windows built-in voice, one WAV per shot.
# Reads out/demo/voice/narration.json in the WSL project (written by
# tools/demo_video.py --voice-text) and writes <shot>.wav next to it.
# Then tools/demo_video.py --mux puts the voice under the video.
#
#   powershell -ExecutionPolicy Bypass -File tools\demo_voice.ps1 [-Voice "Microsoft David Desktop"] [-Rate 0]
param(
  [string]$Dir = "\\wsl.localhost\Ubuntu-24.04\home\joonse\graspgrade\out\demo\voice",
  [string]$Voice = "Microsoft David Desktop",
  [int]$Rate = 0
)
Add-Type -AssemblyName System.Speech
$text = Get-Content -Raw -Encoding UTF8 (Join-Path $Dir "narration.json") | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice($Voice)
$s.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($p in $text.PSObject.Properties) {
  $t = $p.Value
  $out = Join-Path $Dir ($p.Name + ".wav")
  $s.SetOutputToWaveFile($out, $fmt)
  $s.Speak($t)
  $s.SetOutputToNull()
  Write-Output $p.Name
}
$s.Dispose()
