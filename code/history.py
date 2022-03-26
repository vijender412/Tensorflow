import matplotlib.pyplot as plt
import datetime

__all__ = ["plot_history"]


def plot_history(history, individual_figsize=(7, 4), return_figure=False, **kwargs):
    """
    Plot the training history of one or more models.
    This creates a column of plots, with one plot for each metric recorded during training, with the
    plot showing the metric vs. epoch. If multiple models have been trained (that is, a list of
    histories is passed in), each metric plot includes multiple train and validation series.
    Validation data is optional (it is detected by metrics with names starting with ``val_``).
    Args:
        history: the training history, as returned by :meth:`tf.keras.Model.fit`
        individual_figsize (tuple of numbers): the size of the plot for each metric
        return_figure (bool): if True, then the figure object with the plots is returned, None otherwise.
        kwargs: additional arguments to pass to :meth:`matplotlib.pyplot.subplots`
    Returns:
        :class:`matplotlib.figure.Figure`: The figure object with the plots if ``return_figure=True``, None otherwise
    """

    # explicit colours are needed if there's multiple train or multiple validation series, because
    # each train series should have the same color. This uses the global matplotlib defaults that
    # would be used for a single train and validation series.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_train = colors[0]
    color_validation = colors[1]

    if not isinstance(history, list):
        history = [history]

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]

    metrics = sorted({remove_prefix(m, "val_") for m in history[0].history.keys()})

    width, height = individual_figsize
    overall_figsize = (width, len(metrics) * height)

    # plot each metric in a column, so that epochs are aligned (squeeze=False, so we don't have to
    # special case len(metrics) == 1 in the zip)
    fig, all_axes = plt.subplots(
        len(metrics), 1, squeeze=False, sharex="col", figsize=overall_figsize, **kwargs
    )

    has_validation = False
    for ax, m in zip(all_axes[:, 0], metrics):
        for h in history:
            # summarize history for metric m
            ax.plot(h.history[m], c=color_train)

            try:
                val = h.history["val_" + m]
            except KeyError:
                # no validation data for this metric
                pass
            else:
                ax.plot(val, c=color_validation)
                has_validation = True

        ax.set_ylabel(m, fontsize="x-large")

    # don't be redundant: only include legend on the top plot
    labels = ["train"]
    if has_validation:
        labels.append("validation")
    all_axes[0, 0].legend(labels, loc="best", fontsize="x-large")

    # ... and only label "epoch" on the bottom
    all_axes[-1, 0].set_xlabel("epoch", fontsize="x-large")

    # minimise whitespace
    fig.tight_layout()

    if return_figure:
        return fig


def save_history(hist, save_name):
    """ This function saves the history returned by model.fit to a tab-	delimited file, where model is a keras model"""

    # Open file
    fid = open(f'{save_name}', 'a')
    print('trained at {}'.format(datetime.datetime.utcnow()))

    metrics_list = list(hist.history.keys())

    mystr = "iteration\t"
    mystr += "\t".join(metrics_list)

    print(mystr, file=fid)

    try:
        # Iterate through
        for i in range(len(hist.history['loss'])):

            mystr = str(i+1)+'\t'
            for j in metrics_list:
                mystr += str(hist.history[j][i])+'\t'

            print(mystr, file=fid)
            # print('{}\t{}\t{}\t{}\t{}'.format(i + 1,
            #                                   hist.history['batch'][i],
            #                                   hist.history['size'][i],
            #                                   hist.history['loss'][i],
            #                                   hist.history['val_loss'][i],
            #       file=fid)
    except KeyError:
        print('<no history found or error in saving history>', file=fid)

    # Close file
    fid.close()
    return True
