// Client side of `plan_sheet.PlanSheet`: a Jspreadsheet CE (MIT) sheet, for
// Excel-style range selection, copy/paste (TSV, to/from Excel and Google
// Sheets) and the drag-to-fill handle -- all AG Grid Enterprise-only.
//
// The server owns the rows: every edit (typed, pasted or filled) goes up as
// one `cells_changed` event, and the server's reload comes back down as the
// `rows` prop. Column 0 is each row's hidden command id; a row without one
// (the spare last row, or one a paste added) is a new command.

import jspreadsheet from "https://cdn.jsdelivr.net/npm/jspreadsheet-ce@5.0.4/+esm";

// Copies `rows`, never empty: Jspreadsheet's `minSpareRows` crashes on no data.
const sheetData = (rows, width) =>
  rows.length ? rows.map((r) => [...r]) : [Array(width).fill("")];

export default {
  template: "<div></div>",
  props: { columns: Array, rows: Array },

  mounted() {
    this.ws = jspreadsheet(this.$el, {
      worksheets: [
        {
          data: sheetData(this.rows, this.columns.length),
          columns: this.columns.map((c) => ({
            align: "left",
            ...c,
            // Warnings (`check`) in red.
            ...(c.name === "check" && {
              render: (td, value) => td.classList.toggle("plan-check", !!value),
            }),
          })),
          minSpareRows: 1, // Typing in the last, blank row adds a command.
          allowInsertColumn: false,
          allowDeleteColumn: false,
          allowRenameColumn: false,
          allowManualInsertRow: false,
          allowDeleteRow: false,
          columnDrag: false,
          rowDrag: false,
          columnSorting: false,
          parseFormulas: false, // A cell starting with "=" is just text.
          tableOverflow: true,
          tableHeight: "65vh",
          tableWidth: "100%",
        },
      ],
      // Clicking a button outside the sheet clears its selection, so keep
      // the last one for `selectedIds`.
      onselection: (ws, x1, y1, x2, y2) => {
        this.selectedRows = [Math.min(y1, y2), Math.max(y1, y2)];
      },
      // Apply any server update held back while this cell was being edited
      // -- after the edit's own change event, once its editor's gone.
      oneditionend: () => setTimeout(() => this.applyPending()),
      // Its default menu inserts/deletes rows behind the server's back.
      contextMenu: () => false,
      onafterchanges: (ws, changes) => {
        const data = ws.options.data;
        const out = changes
          .filter((c) => Number(c.x) !== 0)
          .map((c) => ({
            id: data[Number(c.y)][0] || null,
            row: Number(c.y),
            field: this.columns[Number(c.x)].name,
            value: c.value == null ? "" : String(c.value),
          }));
        if (out.length) this.$emit("cells_changed", out);
      },
    })[0];
    this.widths = [];
    this.autosize();
  },

  unmounted() {
    jspreadsheet.destroy(this.$el);
  },

  watch: {
    rows(rows) {
      this.pendingRows = rows;
      // Rebuilding under an open editor would throw away what's being typed.
      if (!this.ws.edition) this.applyPending();
    },
  },

  methods: {
    // Widen each column to fit its widest cell (or header), up to its
    // `width` -- never narrowing it, so it doesn't jump about as rows change.
    autosize() {
      const data = this.ws.options.data;
      const ctx = document.createElement("canvas").getContext("2d");
      ctx.font = getComputedStyle(this.$el.querySelector(".jss_worksheet tbody td") ?? this.$el).font;
      const textWidth = (text) => ctx.measureText(String(text ?? "")).width;
      this.columns.forEach((col, x) => {
        if (col.type === "hidden") return;
        const widest = Math.max(textWidth(col.title), ...data.map((r) => textWidth(r[x])));
        const padding = col.type === "dropdown" ? 40 : 16; // Room for its arrow.
        const fit = Math.round(Math.min(Math.max(widest + padding, 50), col.width));
        if (fit > (this.widths[x] ?? 0)) this.ws.setWidth(x, (this.widths[x] = fit));
      });
    },

    applyPending() {
      const rows = this.pendingRows;
      if (!rows || this.ws.edition) return;
      this.pendingRows = null;
      const ws = this.ws;
      const current = ws.options.data.filter((r) => r[0]);
      ws.parent.ignoreEvents = true; // Not the user's edits: don't echo them.
      try {
        const sameRows =
          rows.length === current.length &&
          rows.every((r, y) => String(r[0]) === String(current[y][0]));
        if (!sameRows) {
          this.selectedRows = null; // Its rows may have moved.
          ws.setData(sheetData(rows, this.columns.length));
          return;
        }
        // Same rows: only touch changed cells, leaving the selection alone.
        rows.forEach((r, y) =>
          r.forEach((value, x) => {
            if (String(current[y][x] ?? "") !== String(value ?? "")) {
              ws.setValueFromCoords(x, y, value, true);
            }
          }),
        );
      } finally {
        ws.parent.ignoreEvents = false;
        this.autosize();
      }
    },

    selectedIds() {
      if (!this.selectedRows) return [];
      const [first, last] = this.selectedRows;
      return this.ws.options.data
        .slice(first, last + 1)
        .map((r) => Number(r[0]))
        .filter(Boolean);
    },
  },
};
