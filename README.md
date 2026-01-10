# AIA

## Documentation
https://www.football-data.org/documentation/quickstart

## API
http://api.football-data.org/v4/

## Environment
- Copy `env.example` to `.env` (or use `env`) and set `FOOTBALL_DATA_API_TOKEN`.

## Setup (run the notebook)
1. Create & activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install requests jupyter
```

3. Create `.env` and add your token:

```bash
cp env.example .env
```

Edit `.env` and set:

- `FOOTBALL_DATA_API_TOKEN=...`

4. Start Jupyter and run the notebook:

```bash
jupyter notebook
```

Open `code/parser.ipynb` and run the cells from top to bottom.

### Endpoints

| (Sub)Resource | Action | URI | Filters |
|---|---|---|---|
| Area | List one particular area. | `/v4/areas/{id}` | - |
| Areas | List all available areas. | `/v4/areas/` | - |
| Competition | List one particular competition. | `/v4/competitions/PL` | - |
| Competition | List all available competitions. | `/v4/competitions/` | `areas={AREAS}` |
| Competition / Standings | Show Standings for a particular competition. | `/v4/competitions/{id}/standings` | `matchday={MATCHDAY}`<br>`season={YEAR}`<br>`date={DATE}` |
| Competition / Match | List all matches for a particular competition. | `/v4/competitions/{id}/matches` | `dateFrom={DATE}`<br>`dateTo={DATE}`<br>`stage={STAGE}`<br>`status={STATUS}`<br>`matchday={MATCHDAY}`<br>`group={GROUP}`<br>`season={YEAR}` |
| Competition / Teams | List all teams for a particular competition. | `/v4/competitions/{id}/teams` | `season={YEAR}` |
| Competition / (Top)Scorers | List top scorers for a particular competition. | `/v4/competitions/{id}/scorers` | `limit={LIMIT}`<br>`season={YEAR}` |
| Team | Show one particular team. | `/v4/teams/{id}` | - |
| Team | List teams. | `/v4/teams/` | `limit={LIMIT}`<br>`offset={OFFSET}` |
| Match | Show all matches for a particular team. | `/v4/teams/{id}/matches/` | `dateFrom={DATE}`<br>`dateTo={DATE}`<br>`season={YEAR}`<br>`competitions={competitionIds}`<br>`status={STATUS}`<br>`venue={VENUE}`<br>`limit={LIMIT}` |
| Person | List one particular person. | `/v4/persons/{id}` | - |
| Person / Match | Show all matches for a particular person. | `/v4/persons/{id}/matches` | `dateFrom={DATE}`<br>`dateTo={DATE}`<br>`status={STATUS}`<br>`competitions={competitionIds}`<br>`limit={LIMIT}`<br>`offset={OFFSET}` |
| Match | Show one particular match. | `/v4/matches/{id}` | - |
| Match | List matches across (a set of) competitions. | `/v4/matches` | `competitions={competitionIds}`<br>`ids={matchIds}`<br>`dateFrom={DATE}`<br>`dateTo={DATE}`<br>`status={STATUS}` |
| Match / Head2Head | List previous encounters for the teams of a match. | `/v4/matches/{id}/head2head` | `limit={LIMIT}`<br>`dateFrom={DATE}`<br>`dateTo={DATE}`<br>`competitions={competitionIds}` |

### Filters

| Filter | Type | Description / Possible values |
|---|---|---|
| `id` | Integer `/[0-9]+/` | The id of a resource. |
| `ids` | Integer `/[0-9]+/` | Comma separated list of ids. |
| `matchday` | Integer `/[1-4]+[0-9]*/` | - |
| `season` | String `/yyyy/` | The starting year of a season e.g. 2017 or 2016 |
| `status` | Enum `/[A-Z]+/` | The status of a match. `[SCHEDULED \| LIVE \| IN_PLAY \| PAUSED \| FINISHED \| POSTPONED \| SUSPENDED \| CANCELLED]` |
| `venue` | Enum `/[A-Z]+/` | Defines the venue (type). `[HOME \| AWAY]` |
| `date` / `dateFrom` / `dateTo` | String `/yyyy-MM-dd/` | e.g. 2018-06-22 |
| `stage` | Enum `/[A-Z]+/` | `FINAL` \| `THIRD_PLACE` \| `SEMI_FINALS` \| `QUARTER_FINALS` \| `LAST_16` \| `LAST_32` \| `LAST_64` \| `ROUND_4` \| `ROUND_3` \| `ROUND_2` \| `ROUND_1` \| `GROUP_STAGE` \| `PRELIMINARY_ROUND` \| `QUALIFICATION` \| `QUALIFICATION_ROUND_1` \| `QUALIFICATION_ROUND_2` \| `QUALIFICATION_ROUND_3` \| `PLAYOFF_ROUND_1` \| `PLAYOFF_ROUND_2` \| `PLAYOFFS` \| `REGULAR_SEASON` \| `CLAUSURA` \| `APERTURA` \| `CHAMPIONSHIP` \| `RELEGATION` \| `RELEGATION_ROUND` |
| `plan` | String `/[A-Z]+/` | `TIER_ONE` \| `TIER_TWO` \| `TIER_THREE` \| `TIER_FOUR` |
| `competitions` | String `/\d+,\d+/` | Comma separated list of competition ids. |
| `areas` | String `/\d+,\d+/` | Comma separated list of area ids. |
| `group` | String `/[A-Z_]+/` | Allows filtering for groupings in a competition. |
| `limit` | Integer `/\d+/` | Limits your result set to the given number. Defaults to 10. |
| `offset` | Integer `/\d+/` | Skip offset no. of records when using a limit to page the result list. |
