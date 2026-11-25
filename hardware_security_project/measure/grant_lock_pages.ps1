# Grant "Lock pages in memory" (SeLockMemoryPrivilege) to a user, so the cache
# measurement engine can allocate 2 MiB large pages (needed for a reliable L2
# eviction-set finder on this many-set L2).
#
# Works on Windows Home (uses the LSA API directly; no secpol.msc needed).
# RUN IN AN ELEVATED PowerShell (Run as Administrator), then LOG OFF/ON or REBOOT
# for the privilege to take effect. Reversible: see grant_lock_pages.ps1 -Remove.
#
#   powershell -ExecutionPolicy Bypass -File measure\grant_lock_pages.ps1
#   powershell -ExecutionPolicy Bypass -File measure\grant_lock_pages.ps1 -Remove
param(
  [string]$Account = "$env:USERNAME",
  [switch]$Remove
)

$def = @'
using System;
using System.Runtime.InteropServices;
public class LsaRights {
  [StructLayout(LayoutKind.Sequential)]
  struct LSA_UNICODE_STRING { public ushort Length; public ushort MaximumLength; public IntPtr Buffer; }
  [StructLayout(LayoutKind.Sequential)]
  struct LSA_OBJECT_ATTRIBUTES { public int Length; public IntPtr RootDirectory; public IntPtr ObjectName;
    public int Attributes; public IntPtr SecurityDescriptor; public IntPtr SecurityQualityOfService; }
  [DllImport("advapi32.dll", SetLastError=true)]
  static extern uint LsaOpenPolicy(IntPtr sys, ref LSA_OBJECT_ATTRIBUTES oa, int access, out IntPtr h);
  [DllImport("advapi32.dll", SetLastError=true)]
  static extern uint LsaAddAccountRights(IntPtr h, byte[] sid, LSA_UNICODE_STRING[] rights, int count);
  [DllImport("advapi32.dll", SetLastError=true)]
  static extern uint LsaRemoveAccountRights(IntPtr h, byte[] sid, bool all, LSA_UNICODE_STRING[] rights, int count);
  [DllImport("advapi32.dll")] static extern uint LsaClose(IntPtr h);
  [DllImport("advapi32.dll")] static extern int LsaNtStatusToWinError(uint status);
  static LSA_UNICODE_STRING Str(string s) {
    var u = new LSA_UNICODE_STRING();
    u.Buffer = Marshal.StringToHGlobalUni(s);
    u.Length = (ushort)(s.Length * 2);
    u.MaximumLength = (ushort)(u.Length + 2);
    return u;
  }
  static IntPtr Open() {
    var oa = new LSA_OBJECT_ATTRIBUTES(); IntPtr h;
    uint st = LsaOpenPolicy(IntPtr.Zero, ref oa, 0x00000810, out h);  // CREATE_ACCOUNT | LOOKUP_NAMES
    if (st != 0) throw new Exception("LsaOpenPolicy err " + LsaNtStatusToWinError(st));
    return h;
  }
  public static void Add(byte[] sid, string right) {
    IntPtr h = Open();
    uint st = LsaAddAccountRights(h, sid, new LSA_UNICODE_STRING[]{ Str(right) }, 1);
    LsaClose(h);
    if (st != 0) throw new Exception("LsaAddAccountRights err " + LsaNtStatusToWinError(st));
  }
  public static void Remove(byte[] sid, string right) {
    IntPtr h = Open();
    uint st = LsaRemoveAccountRights(h, sid, false, new LSA_UNICODE_STRING[]{ Str(right) }, 1);
    LsaClose(h);
    if (st != 0) throw new Exception("LsaRemoveAccountRights err " + LsaNtStatusToWinError(st));
  }
}
'@
Add-Type -TypeDefinition $def

$sid = (New-Object System.Security.Principal.NTAccount($Account)).Translate([System.Security.Principal.SecurityIdentifier])
$bytes = New-Object byte[] $sid.BinaryLength
$sid.GetBinaryForm($bytes, 0)
$priv = "SeLockMemoryPrivilege"

if ($Remove) {
  [LsaRights]::Remove($bytes, $priv)
  Write-Output "Removed $priv from $Account. Log off/on (or reboot) to apply."
} else {
  [LsaRights]::Add($bytes, $priv)
  Write-Output "Granted $priv to $Account ($($sid.Value))."
  Write-Output "Now LOG OFF/ON or REBOOT, then run the measurement from an ELEVATED terminal."
}
