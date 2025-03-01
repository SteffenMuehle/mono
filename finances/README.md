requirements:
- laptop
- phone+passwords for BLSK + ING app

1. Log into http://www.comdirect.de
    - move Verrechnungskonto to Giro
    - pay off Visacard
2. Log into http://www.paypal.com
    - check diff to 500€, and send money from Comdirect to
    - Empfänger: PayPal (Europe) S.a r.l. et Cie, S.C.A.
    - IBAN: DE97 1207 0088 3037 1176 01
3. Download depot csv in Comdirect, put into finances/data/input/depot, format csv manually
4. Update finances/data/io.toml with major expenses and Liz contributions
5. log into blsk.de
6. log into ing.de
7. `just clean`
8. `just prompt`
9. `just depot`
10. `just graph`
11. git commit stuff in finances/data repo
