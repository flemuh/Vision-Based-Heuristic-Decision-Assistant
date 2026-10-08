SPEEDLORA PORTABLE PROFILE MIGRATION

This folder contains the ONE-TIME migration helper only.
The application source/database format is not changed by this migration.

Current profile source:
  %LOCALAPPDATA%\Speedlora\JewelBingo-V1826-StateIntegrity

Portable destination inside the project:
  <project>\userdata\

Safe flow
---------
1. Close Speedlora completely.
2. Run MIGRATE_EXISTING_PROFILE_TO_PORTABLE.bat from this folder.
3. The BAT COPYs the complete profile to userdata.__migrating__ first.
4. It verifies the copy with robocopy dry-run comparison.
5. Only after verification is the temporary copy renamed to userdata.
6. The original LocalAppData profile is NEVER deleted.
7. Start Speedlora using the project-root RUN_PORTABLE.bat.
8. After a few successful games, this migration folder may be deleted.

Moving to another PC
--------------------
Copy/ZIP the project folder INCLUDING userdata. On the new PC use
RUN_PORTABLE.bat. The profile includes SQLite history, templates,
calibration, learning artifacts, configs, screenshots and runtime logs.

Important
---------
After validating portable mode, do not alternate between run_windows.bat
and RUN_PORTABLE.bat: they point to different profile copies and their
histories will diverge.
