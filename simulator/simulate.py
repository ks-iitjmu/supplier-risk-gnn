"""Procurement network simulator.

Generates the tables an ERP/procurement system would export, with four procurement
fraud schemes (from the ACFE fraud tree) planted among legitimate behaviour, plus
deliberate confounders so the problem is not trivially separable.

Outputs
  data/raw/            what a fraud team would actually have
    suppliers.csv, supplier_directors.csv, buyers.csv, products.csv,
    invoices.csv, shipments.csv, tenders.csv, bids.csv, audit_labels.csv
  data/ground_truth/   hidden answers, used ONLY for final evaluation
    supplier_truth.csv (true fraud flag, scheme, group, canonical address key)

Fraud schemes
  shell_ring   3-6 recently registered suppliers sharing bank accounts, addresses
               or directors; bill a captured buyer for goods that never ship.
               Some members are clean-looking "fronts" -> only links expose them.
  cartel       3-5 established suppliers rigging tenders at one buyer: all bid,
               winners rotate, cover bids sit just above the winner, prices inflated.
  split        invoices split to sit just under the buyer's approval limit.
  inflate      unit prices well above market on part of the invoices.

Confounders
  business parks (legit suppliers sharing a building), corporate groups (legit
  suppliers sharing directors / bank accounts), legit invoices near the approval
  limit, legit short or missing shipment records, service suppliers with no goods
  receipts, premium-priced legit suppliers, legit frequent bidders, legit new suppliers, and
  incomplete audit labels (not all fraud is ever caught).
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["Steel & Metals", "Electrical", "IT Hardware", "Office Supplies",
              "Chemicals", "Packaging", "Industrial Tools", "Safety Equipment",
              "Electronic Components", "Furniture", "Facility Services", "Logistics"]
SERVICE_CATEGORIES = ["Facility Services", "Logistics"]
CITIES = {"Bengaluru": ("Bangalore", 560000), "Mumbai": ("Bombay", 400000),
          "Pune": ("Poona", 411000), "New Delhi": ("Delhi", 110000),
          "Chennai": ("Madras", 600000), "Hyderabad": ("Hyderabad", 500000),
          "Ahmedabad": ("Amdavad", 380000), "Kolkata": ("Calcutta", 700000),
          "Jammu": ("Jammu", 180000), "Gurugram": ("Gurgaon", 122000)}
STREETS = ["MG Road", "Station Road", "Ring Road", "Industrial Area Phase 2",
           "Nehru Nagar Main Road", "Gandhi Marg", "Sector 18", "Link Road",
           "Old Airport Road", "GIDC Estate Phase 1", "Lake View Road", "Sector 44"]
NAME_A = ["Shree", "Sai", "Om", "Bharat", "Ganesh", "Kaveri", "Sunrise", "Apex", "Vertex",
          "Neel", "Trident", "Orient", "Pioneer", "Galaxy", "Krishna", "Lakshmi",
          "Maruti", "Indus", "Zenith", "Royal", "Everest", "Sapphire", "Unity", "Prime"]
NAME_B = ["Steel", "Electricals", "Tech", "Enterprises", "Industries", "Traders",
          "Solutions", "Packaging", "Chemicals", "Logistics", "Infra", "Systems", "Supplies"]
NAME_C = ["Pvt Ltd", "Private Limited", "LLP", "& Co", "Ltd"]
FIRST = ["Rahul", "Amit", "Priya", "Neha", "Vikram", "Anjali", "Suresh", "Kavita", "Rohit",
         "Pooja", "Arjun", "Meera", "Sanjay", "Divya", "Manoj", "Sunita", "Karan", "Ritu",
         "Deepak", "Asha", "Nikhil", "Shreya", "Rajesh", "Swati"]
LAST = ["Sharma", "Verma", "Gupta", "Iyer", "Reddy", "Patel", "Singh", "Nair", "Mehta",
        "Joshi", "Rao", "Kapoor", "Das", "Kulkarni", "Banerjee", "Chopra", "Malhotra", "Bhat"]


@dataclass
class SimConfig:
    n_suppliers: int = 5000
    n_buyers: int = 40
    products_per_category: int = 25
    start: str = "2024-01-01"
    days: int = 730
    seed: int = 7
    n_shell_rings: int = 30
    n_cartels: int = 15
    frac_split: float = 0.012
    frac_inflate: float = 0.012
    n_business_parks: int = 60
    frac_corporate_group: float = 0.06
    frac_legit_near_limit: float = 0.03
    frac_legit_new: float = 0.08


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = pd.Timestamp(cfg.start)
        self.t1 = self.t0 + pd.Timedelta(days=cfg.days)

    # ------------------------------------------------------------------ helpers
    def _date(self, lo: pd.Timestamp, hi: pd.Timestamp, n: int) -> pd.DatetimeIndex:
        span = max(int((hi - lo).days), 1)
        return lo + pd.to_timedelta(self.rng.integers(0, span, n), unit="D")

    def _new_address(self, unit=None) -> dict:
        r = self.rng
        city = r.choice(list(CITIES))
        return {"unit": unit, "no": int(r.integers(1, 400)), "street": r.choice(STREETS),
                "city": city, "pin": CITIES[city][1] + int(r.integers(1, 99))}

    def _render(self, a: dict) -> str:
        """Same place, many spellings - forces entity resolution downstream."""
        r = self.rng
        street = a["street"]
        if r.random() < 0.5:
            street = (street.replace("Road", r.choice(["Rd", "Rd."]))
                      .replace("Sector", r.choice(["Sec", "Sec."]))
                      .replace("Phase", r.choice(["Ph", "Ph."]))
                      .replace("MG", r.choice(["M.G.", "M G", "Mahatma Gandhi"])))
        no = r.choice([f"{a['no']},", f"No. {a['no']},", f"#{a['no']}", f"{a['no']}"])
        alt, _ = CITIES[a["city"]]
        city = alt if r.random() < 0.3 else a["city"]
        pin = str(a["pin"])
        pin = r.choice([pin, f"{pin[:3]} {pin[3:]}", f"- {pin}"])
        unit = ""
        if a["unit"]:
            unit = r.choice([f"Unit {a['unit']}, ", f"{a['unit']}, ", f"Office {a['unit']}, "])
        s = f"{unit}{no} {street}, {city} {pin}"
        return s.upper() if r.random() < 0.15 else s

    def _addr_key(self, a: dict) -> str:
        return f"{a['no']}|{a['street']}|{a['pin']}|{a['unit'] or ''}"

    # ------------------------------------------------------------- master data
    def make_products(self):
        rows = []
        for ci, cat in enumerate(CATEGORIES):
            base = self.rng.lognormal(np.log(2000), 0.9)
            for j in range(self.cfg.products_per_category):
                rows.append({"product_id": f"P{ci:02d}{j:03d}", "category": cat,
                             "market_unit_price": round(base * self.rng.lognormal(0, 0.6), 2)})
        self.products = pd.DataFrame(rows)
        self.cat_products = {c: g["product_id"].values for c, g in self.products.groupby("category")}
        self.market = self.products.set_index("product_id")["market_unit_price"]

    def make_buyers(self):
        r = self.rng
        n = self.cfg.n_buyers
        self.buyers = pd.DataFrame({
            "buyer_id": [f"B{i:03d}" for i in range(n)],
            "business_unit": [f"{r.choice(['Plant', 'Warehouse', 'Corporate', 'R&D', 'Projects'])}-"
                              f"{r.choice(list(CITIES))}" for _ in range(n)],
            "approval_limit": r.choice([200_000, 500_000, 1_000_000], n, p=[.4, .4, .2]),
        })
        self.limit = self.buyers.set_index("buyer_id")["approval_limit"]
        self.buyer_weights = r.dirichlet(np.ones(n) * 2)

    def make_suppliers(self):
        r, c = self.rng, self.cfg
        n = c.n_suppliers
        old = self._date(pd.Timestamp("2005-01-01"), pd.Timestamp("2023-06-30"), n)
        s = pd.DataFrame({
            "supplier_id": [f"S{i:05d}" for i in range(n)],
            "supplier_name": [f"{r.choice(NAME_A)} {r.choice(NAME_B)} {r.choice(NAME_C)}"
                              for _ in range(n)],
            "category": r.choice(CATEGORIES, n),
            "registration_date": old,
            "rate": r.lognormal(np.log(15), 0.7, n),       # invoices per year
            "markup": np.where(r.random(n) < 0.06, r.uniform(0.12, 0.30, n),  # premium brands
                               r.normal(0.02, 0.05, n)),
            "bid_propensity": r.lognormal(0, 1.3, n),             # some vendors bid a lot
            "scheme": "none", "group_id": "",
        })
        new = r.random(n) < c.frac_legit_new                # confounder: legit new suppliers
        s.loc[new, "registration_date"] = self._date(self.t0 - pd.Timedelta(days=200),
                                                     self.t0 + pd.Timedelta(days=300), new.sum())
        s["buyers"] = [list(r.choice(self.buyers["buyer_id"], min(1 + r.poisson(1), 5),
                                     replace=False, p=self.buyer_weights)) for _ in range(n)]
        s["address"] = [self._new_address() for _ in range(n)]
        s["bank_account"] = [f"ACCT-{r.integers(10**9, 10**10)}" for _ in range(n)]
        self.next_din = 10_000_000
        s["directors"] = [[self._new_director() for _ in range(r.integers(1, 4))] for _ in range(n)]
        s["near_limit"] = False
        s["phantom_rate"] = 0.0
        s["complicit_buyer"] = None
        self.s = s

    def _new_director(self):
        self.next_din += int(self.rng.integers(1, 50))
        return (self.next_din, f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}")

    def _take(self, pool: set, k: int, mask=None) -> list:
        cand = [i for i in pool if mask is None or mask[i]]
        pick = list(self.rng.choice(cand, k, replace=False))
        pool.difference_update(pick)
        return pick

    # ---------------------------------------------------------- confounders
    def add_confounders(self):
        r, c, s = self.rng, self.cfg, self.s
        free = set(range(len(s)))
        for p in range(c.n_business_parks):
            building = self._new_address()
            for i in self._take(free, int(r.integers(4, 11))):
                a = dict(building)
                a["unit"] = f"{r.integers(1, 9)}{r.choice(list('ABCDEF'))}"
                s.at[i, "address"] = a
        free = set(range(len(s)))
        n_grouped = int(c.frac_corporate_group * len(s))
        while n_grouped > 0:
            members = self._take(free, int(r.integers(2, 5)))
            shared = self._new_director()
            acct = s.at[members[0], "bank_account"] if r.random() < 0.25 else None
            for i in members:
                s.at[i, "directors"] = s.at[i, "directors"] + [shared]
                if acct:
                    s.at[i, "bank_account"] = acct
            n_grouped -= len(members)
        s["near_limit"] = r.random(len(s)) < c.frac_legit_near_limit

    # --------------------------------------------------------- fraud schemes
    def plant_fraud(self):
        r, c, s = self.rng, self.cfg, self.s
        pool = set(range(len(s)))
        self.cartels = []
        established = (s["registration_date"] < pd.Timestamp("2022-01-01")).values

        for g in range(c.n_shell_rings):
            members = self._take(pool, int(r.integers(3, 7)))
            buyer = r.choice(self.buyers["buyer_id"], p=self.buyer_weights)
            links = [k for k in ["bank", "address", "director"] if r.random() < 0.55] or ["director"]
            addr = self._new_address(unit=f"{r.integers(1, 20)}{r.choice(list('ABC'))}")
            acct = f"ACCT-{r.integers(10**9, 10**10)}"
            director = self._new_director()
            for i in members:
                s.at[i, "scheme"], s.at[i, "group_id"] = "shell_ring", f"R{g:02d}"
                s.at[i, "registration_date"] = self._date(self.t0 - pd.Timedelta(days=250),
                                                          self.t0 + pd.Timedelta(days=250), 1)[0]
                s.at[i, "buyers"] = [buyer] + list(r.choice(self.buyers["buyer_id"], 1))
                s.at[i, "rate"] = r.lognormal(np.log(12), 0.5)
                # fronts look clean (phantom_rate 0); others bill for missing goods
                s.at[i, "phantom_rate"] = 0.0 if r.random() < 0.35 else r.uniform(0.25, 0.7)
                if "address" in links and r.random() < 0.85:
                    s.at[i, "address"] = dict(addr)
                if "bank" in links and r.random() < 0.85:
                    s.at[i, "bank_account"] = acct
                if "director" in links and r.random() < 0.85:
                    s.at[i, "directors"] = [director] + s.at[i, "directors"][:1]

        for g in range(c.n_cartels):
            cat = r.choice(CATEGORIES)
            mask = established & (s["category"] == cat).values
            members = self._take(pool, int(r.integers(3, 6)), mask)
            buyer = r.choice(self.buyers["buyer_id"], p=self.buyer_weights)
            for i in members:
                s.at[i, "scheme"], s.at[i, "group_id"] = "cartel", f"C{g:02d}"
                if buyer not in s.at[i, "buyers"]:
                    s.at[i, "buyers"] = s.at[i, "buyers"] + [buyer]
            self.cartels.append((f"C{g:02d}", buyer, cat, members))

        for scheme, frac in [("split", c.frac_split), ("inflate", c.frac_inflate)]:
            for i in self._take(pool, int(frac * len(s))):
                s.at[i, "scheme"] = scheme
                if scheme == "split":
                    s.at[i, "complicit_buyer"] = r.choice(self.buyers["buyer_id"])

    # -------------------------------------------------------------- invoices
    def make_invoices(self):
        r, s = self.rng, self.s
        cartel_buyer = {i: b for _, b, _, m in self.cartels for i in m}
        rows = []
        for i, sup in s.iterrows():
            lo = max(sup["registration_date"], self.t0)
            n = r.poisson(sup["rate"] * (self.t1 - lo).days / 365)
            if n == 0:
                continue
            dates = self._date(lo, self.t1, n)
            buyers = list(sup["buyers"])
            p = None
            if sup["scheme"] == "shell_ring":
                p = [0.8] + [0.2 / (len(buyers) - 1)] * (len(buyers) - 1)
            elif sup["scheme"] == "split":
                buyers = buyers + [sup["complicit_buyer"]]
                p = [0.5 / (len(buyers) - 1)] * (len(buyers) - 1) + [0.5]
            b = r.choice(buyers, n, p=p)
            prod = r.choice(self.cat_products[sup["category"]], n)
            mkt = self.market[prod].values
            price = mkt * (1 + sup["markup"] + r.normal(0, 0.06, n))
            qty = np.maximum(1, np.round(r.lognormal(np.log(40), 1.0, n)))
            lim = self.limit[b].values

            if sup["scheme"] == "inflate":
                hit = r.random(n) < r.uniform(0.3, 0.7)
                price = np.where(hit, price * r.uniform(1.25, 1.6, n), price)
            if sup["scheme"] == "cartel":
                hit = b == cartel_buyer[i]
                price = np.where(hit, price * r.uniform(1.10, 1.25, n), price)
            if sup["near_limit"]:  # legit contract ceilings near the limit
                hit = r.random(n) < 0.2
                qty = np.where(hit, np.maximum(1, np.round(lim * r.uniform(0.90, 0.99, n) / price)), qty)

            phantom = r.random(n) < sup["phantom_rate"]
            for k in range(n):
                rows.append((sup["supplier_id"], b[k], dates[k], prod[k], qty[k],
                             round(price[k], 2), phantom[k]))

            if sup["scheme"] == "split":  # bursts of just-under-limit invoices
                cb = sup["complicit_buyer"]
                L = self.limit[cb]
                for _ in range(r.poisson(max(1, n * 0.25))):
                    d0 = self._date(lo, self.t1, 1)[0]
                    total, done = L * r.uniform(1.3, 3.5), 0.0
                    pr = r.choice(self.cat_products[sup["category"]])
                    unit = self.market[pr] * (1 + sup["markup"])
                    while done < total:
                        amt = L * r.uniform(0.88, 0.99)
                        q = max(1, np.floor(amt / unit))
                        d = d0 + pd.Timedelta(days=int(r.integers(0, 3)))
                        rows.append((sup["supplier_id"], cb, d, pr, q, round(unit, 2), False))
                        done += q * unit

        inv = pd.DataFrame(rows, columns=["supplier_id", "buyer_id", "invoice_date", "product_id",
                                          "quantity", "unit_price", "_phantom"])
        inv = inv.sort_values("invoice_date").reset_index(drop=True)
        inv.insert(0, "invoice_id", [f"INV{k:07d}" for k in range(len(inv))])
        inv["amount"] = (inv["quantity"] * inv["unit_price"]).round(2)
        self.inv = inv

    def make_shipments(self):
        r, inv = self.rng, self.inv
        n = len(inv)
        u = r.random(n)
        ph = inv["_phantom"].values
        cat = inv["supplier_id"].map(self.s.set_index("supplier_id")["category"])
        service = cat.isin(SERVICE_CATEGORIES).values
        # legit: 1.5% missing records, 2.5% short deliveries (confounders);
        # services have no goods receipt at all
        missing = np.where(ph, u < 0.85, u < 0.015) | (service & (r.random(n) < 0.92))
        short = np.where(ph, u >= 0.85, (u >= 0.015) & (u < 0.04))
        frac = np.where(short, np.where(ph, r.uniform(0, 0.2, n), r.uniform(0.6, 0.95, n)), 1.0)
        ship = pd.DataFrame({
            "invoice_id": inv["invoice_id"],
            "ship_date": inv["invoice_date"] - pd.to_timedelta(r.integers(0, 8, n), unit="D"),
            "quantity_received": np.floor(inv["quantity"].values * frac),
        })[~missing]
        ship.insert(0, "shipment_id", [f"SH{k:07d}" for k in range(len(ship))])
        self.ship = ship
        self.inv = inv.drop(columns="_phantom")

    # --------------------------------------------------------------- tenders
    def make_tenders(self):
        r, s = self.rng, self.s
        by_cat = {c: g.index.values for c, g in s.groupby("category")}
        reg = s["registration_date"].values
        tenders, bids = [], []

        def add_tender(buyer, cat, date, est, bidders_bids):
            tid = f"T{len(tenders):05d}"
            tenders.append((tid, buyer, cat, date, round(est, 2)))
            win = min(bidders_bids, key=lambda x: x[1])[0]
            for sid, amt in bidders_bids:
                bids.append((tid, s.at[sid, "supplier_id"], round(amt, 2), sid == win))

        for buyer in self.buyers["buyer_id"]:
            for date in self._date(self.t0, self.t1, int(r.integers(30, 70))):
                cat = r.choice(CATEGORIES)
                cand = by_cat[cat][reg[by_cat[cat]] <= np.datetime64(date)]
                k = min(len(cand), int(r.integers(3, 7)))
                est = r.lognormal(np.log(800_000), 0.6)
                w = s.loc[cand, "bid_propensity"].values
                pick = r.choice(cand, k, replace=False, p=w / w.sum())
                add_tender(buyer, cat, date, est,
                           [(i, est * r.uniform(0.88, 1.12) * (1 + s.at[i, "markup"])) for i in pick])

        for gid, buyer, cat, members in self.cartels:
            outsiders = np.setdiff1d(by_cat[cat], members)
            for t, date in enumerate(sorted(self._date(self.t0, self.t1, int(r.integers(15, 30))))):
                est = r.lognormal(np.log(800_000), 0.6)
                winner = members[t % len(members)]                  # rotation
                win_bid = est * r.uniform(1.10, 1.25)
                bb = [(winner, win_bid)] + [(i, win_bid * r.uniform(1.03, 1.15))
                                            for i in members if i != winner]  # cover bids
                if r.random() < 0.3:                                 # occasional honest rival
                    for i in r.choice(outsiders, int(r.integers(1, 3)), replace=False):
                        bb.append((i, est * r.uniform(0.95, 1.2) * (1 + s.at[i, "markup"])))
                add_tender(buyer, cat, date, est, bb)

        self.tenders = pd.DataFrame(tenders, columns=["tender_id", "buyer_id", "category",
                                                      "tender_date", "estimated_value"])
        self.bids = pd.DataFrame(bids, columns=["tender_id", "supplier_id", "bid_amount",
                                                "is_winner"])

    # ---------------------------------------------------------------- labels
    def make_labels(self):
        """Audits catch only part of the fraud; uncaught fraud looks like a 0."""
        r, s = self.rng, self.s
        caught = np.zeros(len(s), dtype=bool)
        for gid, g in s[s["group_id"] != ""].groupby("group_id"):
            if r.random() < 0.6:                       # group investigated
                caught[g.index.values] = r.random(len(g)) < 0.7
        solo = s["scheme"].isin(["split", "inflate"]).values
        caught[solo] = r.random(solo.sum()) < 0.6
        self.labels = pd.DataFrame({"supplier_id": s["supplier_id"],
                                    "confirmed_fraud": caught.astype(int)})

    # ---------------------------------------------------------------- export
    def export(self, out: Path):
        s = self.s
        raw, gt = out / "raw", out / "ground_truth"
        raw.mkdir(parents=True, exist_ok=True)
        gt.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "supplier_id": s["supplier_id"], "supplier_name": s["supplier_name"],
            "category": s["category"], "registration_date": s["registration_date"].dt.date,
            "address": [self._render(a) for a in s["address"]],
            "bank_account": s["bank_account"],
        }).to_csv(raw / "suppliers.csv", index=False)
        pd.DataFrame([(sid, din, name) for sid, ds in zip(s["supplier_id"], s["directors"])
                      for din, name in ds],
                     columns=["supplier_id", "director_din", "director_name"]
                     ).to_csv(raw / "supplier_directors.csv", index=False)
        self.buyers.to_csv(raw / "buyers.csv", index=False)
        self.products.to_csv(raw / "products.csv", index=False)
        self.inv.to_csv(raw / "invoices.csv", index=False)
        self.ship.to_csv(raw / "shipments.csv", index=False)
        self.tenders.to_csv(raw / "tenders.csv", index=False)
        self.bids.to_csv(raw / "bids.csv", index=False)
        self.labels.to_csv(raw / "audit_labels.csv", index=False)
        pd.DataFrame({
            "supplier_id": s["supplier_id"], "is_fraud": (s["scheme"] != "none").astype(int),
            "scheme": s["scheme"], "group_id": s["group_id"],
            "address_key": [self._addr_key(a) for a in s["address"]],
        }).to_csv(gt / "supplier_truth.csv", index=False)

    def run(self, out: Path):
        self.make_products()
        self.make_buyers()
        self.make_suppliers()
        self.add_confounders()
        self.plant_fraud()
        self.make_invoices()
        self.make_shipments()
        self.make_tenders()
        self.make_labels()
        self.export(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suppliers", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    Simulator(SimConfig(n_suppliers=a.suppliers, seed=a.seed)).run(Path(a.out))
    print(f"Simulation written to {a.out}")
