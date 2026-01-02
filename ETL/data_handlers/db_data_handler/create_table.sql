-- Run this as superuser (postgres)
CREATE TABLE holdings (
    accessionnumber TEXT,
    infotablesk TEXT,
    nameofissuer TEXT,
    cusip TEXT,
    value NUMERIC,
    sshprnamt BIGINT,
    filingdate DATE,
    cik TEXT,
    value_per_share NUMERIC,
    year INT,
    quarter INT,
    period_start DATE NOT NULL
)
PARTITION BY RANGE (period_start);

GRANT ALL ON holdings TO asaf_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO asaf_user;


ALTER TABLE IF EXISTS public.holdings
    OWNER to postgres;

REVOKE ALL ON TABLE public.holdings FROM asaf_user;

GRANT INSERT, DELETE, SELECT, UPDATE ON TABLE public.holdings TO asaf_user;

GRANT ALL ON TABLE public.holdings TO postgres;

GRANT CREATE ON SCHEMA public TO asaf_user;

CREATE INDEX ON holdings (cusip);
CREATE INDEX ON holdings (cik);
CREATE INDEX ON holdings (filingdate);

