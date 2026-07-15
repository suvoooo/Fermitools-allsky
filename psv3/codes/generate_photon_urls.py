"""
Generates photon_urls.txt and spacecraft_urls.txt for 16 years of Fermi-LAT data.

Week 9  ~ Aug 4, 2008 (start of Fermi science operations)

Verify base URLs from the FSSC download page before running:
https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/LAT_weekly_allsky.html
"""

from datetime import date, timedelta

START_WEEK   = 9
WEEK9_START  = date(2008, 8, 4)   # approximate start date of week 9
TARGET_YEARS = 16

# calculate week range for 16 years
end_date    = date(WEEK9_START.year + TARGET_YEARS, WEEK9_START.month, WEEK9_START.day)
total_days  = (end_date - WEEK9_START).days
total_weeks = total_days // 7
END_WEEK    = START_WEEK + total_weeks  # inclusive: week that straddles the 16-yr mark
n_files     = END_WEEK - START_WEEK + 1

FILES = [
    {
        "label"   : "Photon",
        "base_url": "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon",
        "pattern" : "lat_photon_weekly_w{week:03d}_p305_v001.fits",
        "out"     : "photon_urls.txt",
    },
    {
        "label"   : "Spacecraft",
        "base_url": "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/spacecraft",
        "pattern" : "lat_spacecraft_weekly_w{week:03d}_p310_v001.fits",
        "out"     : "spacecraft_urls.txt",
    },
]

print(f"Fermi-LAT 16-year file range")
print(f"  Start : week {START_WEEK:03d}  ({WEEK9_START})")
print(f"  End   : week {END_WEEK:03d}  (~{WEEK9_START + timedelta(weeks=total_weeks)})")
print(f"  Files : {n_files} per type\n")

for entry in FILES:
    with open(entry["out"], "w") as f:
        for week in range(START_WEEK, END_WEEK + 1):
            filename = entry["pattern"].format(week=week)
            f.write(f"{entry['base_url']}/{filename}\n")
    print(f"{entry['label']:12s} -> {entry['out']}  ({n_files} URLs)")

print("\nVerify the first and last lines of each file before running the wget script.")
