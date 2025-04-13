requirements:
- laptop
- phone+passwords for BLSK + ING app

1. Log into http://www.comdirect.de
    - move Verrechnungskonto to Giro
    - pay off Visacard
    - Download depot csv, put into finances/data/input/depot/comdirect, format csv manually
2. Log into http://www.paypal.com
    - check diff to 500€, and send money from Comdirect to
    - Empfänger: PayPal (Europe) S.a r.l. et Cie, S.C.A.
    - IBAN: DE97 1207 0088 3037 1176 01
3. Log into https://www.ing.de/
    - if not enough on Comdirect Giro, send from Extra-Konto
    - Download depot csv, put into finances/data/input/depot/comdirect, format csv manually
4. Update finances/data/io.toml with major expenses and Liz contributions
5. log into blsk.de
6. log into ing.de
7. `just clean`
8. `just prompt`
9. `just crypto`
10. `just depot`
11. `just graph`
12. git commit stuff in finances/data repo
