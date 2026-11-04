# Record the Try page as it runs, for the clip at the top of the summary page.
# Starts Edge headless with the DevTools protocol, opens the page from a local server,
# picks a clip, follows the steps as they appear, and saves the screencast frames with
# their times. web/tools/try_clip.sh then turns the frames into an mp4.
#
#   powershell -ExecutionPolicy Bypass -File tools\try_record.ps1 -Url "http://localhost:8766/_try_rec.html?lang=en" -Out C:\...\frames_en
param(
  [string]$Url = "http://localhost:8766/_try_rec.html?lang=en",
  [string]$Out = "$env:TEMP\try_frames",
  [int]$Pick = 1,            # which clip card to click
  [double]$Seconds = 19.0,   # length of the take after the page loads
  [int]$Port = 9333
)
$ErrorActionPreference = "Stop"
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
New-Item -ItemType Directory -Force $Out | Out-Null
Get-ChildItem $Out -Filter *.jpg | Remove-Item -Force
$prof = Join-Path $Out "profile"
$p = Start-Process $edge -PassThru -ArgumentList @("--headless=new", "--remote-debugging-port=$Port", "--user-data-dir=$prof",
  "--hide-scrollbars", "--autoplay-policy=no-user-gesture-required", "--window-size=1200,820", "about:blank")
Start-Sleep -Seconds 3
$page = (Invoke-RestMethod "http://127.0.0.1:$Port/json") | Where-Object { $_.type -eq "page" } | Select-Object -First 1
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ws.Options.KeepAliveInterval = [TimeSpan]::FromSeconds(30)
$ws.ConnectAsync([Uri]$page.webSocketDebuggerUrl, [Threading.CancellationToken]::None).Wait()
$script:id = 0
function Send($method, $params) {
  $script:id++
  $msg = @{ id = $script:id; method = $method; params = $params } | ConvertTo-Json -Depth 6 -Compress
  $bytes = [Text.Encoding]::UTF8.GetBytes($msg)
  $ws.SendAsync([ArraySegment[byte]]$bytes, [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).Wait()
}
$buf = New-Object byte[] (1 -shl 20)
$sb = New-Object Text.StringBuilder
$recv = $null
function Next-Message($waitMs) {
  # one full text message, or $null when nothing arrives in time
  if ($script:recv -eq $null) { $script:recv = $ws.ReceiveAsync([ArraySegment[byte]]$buf, [Threading.CancellationToken]::None) }
  if (-not $script:recv.Wait($waitMs)) { return $null }
  $r = $script:recv.Result
  $script:recv = $null
  [void]$sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Count))
  if (-not $r.EndOfMessage) { return Next-Message 5000 }
  $s = $sb.ToString(); [void]$sb.Clear(); return $s
}
function Eval($js) { Send "Runtime.evaluate" @{ expression = $js } }

Send "Emulation.setDeviceMetricsOverride" @{ width = 1200; height = 820; deviceScaleFactor = 1.5; mobile = $false }
Send "Page.enable" @{}
Send "Page.navigate" @{ url = $Url }
$t0 = Get-Date
while (((Get-Date) - $t0).TotalSeconds -lt 3.0) { [void](Next-Message 50) }
Send "Page.startScreencast" @{ format = "jpeg"; quality = 88; everyNthFrame = 1 }
# the script of the take: pause on the finished first clip, click a clip, then keep the newest step in view
$plan = @(
  @{ at = 1.6; js = "document.querySelectorAll('.pick')[$Pick].click()" },
  @{ at = 2.1; js = "window.__follow = setInterval(function () { var s = document.querySelectorAll('.step.on'); var last = s[s.length - 1]; if (last) { var r = last.getBoundingClientRect(); if (r.bottom > innerHeight - 40 || r.top < 0) last.scrollIntoView({ behavior: 'smooth', block: 'center' }); } }, 250)" }
)
$k = 0; $n = 0
$start = Get-Date
$times = New-Object Collections.Generic.List[string]
while ($true) {
  $el = ((Get-Date) - $start).TotalSeconds
  if ($el -ge $Seconds) { break }
  while ($k -lt $plan.Count -and $el -ge $plan[$k].at) { Eval $plan[$k].js; $k++ }
  $m = Next-Message 30
  if ($m -eq $null) { continue }
  if ($m.Contains('"Page.screencastFrame"')) {
    $o = $m | ConvertFrom-Json
    $f = Join-Path $Out ("f{0:D5}.jpg" -f $n)
    [IO.File]::WriteAllBytes($f, [Convert]::FromBase64String($o.params.data))
    $times.Add(("{0:D5} {1}" -f $n, $o.params.metadata.timestamp))
    $n++
    Send "Page.screencastFrameAck" @{ sessionId = $o.params.sessionId }
  }
}
Send "Page.stopScreencast" @{}
[IO.File]::WriteAllLines((Join-Path $Out "times.txt"), $times)
Send "Browser.close" @{}
Start-Sleep -Seconds 1
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
"frames $n"
